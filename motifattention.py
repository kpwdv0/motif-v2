import csv
import json
import math
import os
import tempfile
import time

import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM

PAD_ID = 2  # aria's pad token
MOTIF_LEN = 7  # 6 transitions = 7 notes
MAX_TOKENS = 512


def note_fields(note):
    # melody notes have 3 fields, raw notes have velocity too
    if len(note) > 3:
        return note[0], note[1], note[2], note[3]
    return note[0], note[1], note[2], 80


class MotifCrossAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, motif_vectors, motif_padding_mask=None):
        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=motif_vectors,
            value=motif_vectors,
            key_padding_mask=motif_padding_mask,
        )
        gate = torch.sigmoid(self.gate)
        return hidden_states + gate * attn_out


class BlockWithMotifAttention(nn.Module):

    def __init__(self, original_block, hidden_dim, dropout=0.1):
        super().__init__()
        self.original_block = original_block
        self.motif_attn = MotifCrossAttention(hidden_dim, dropout=dropout)
        self.motif_vectors = None
        self.motif_padding_mask = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = self.original_block(hidden_states, attention_mask, **kwargs)

        if isinstance(out, tuple):
            hidden_states, *extras = out
        else:
            hidden_states = out
            extras = []

        if self.motif_vectors is not None:
            hidden_states = self.motif_attn(hidden_states, self.motif_vectors, self.motif_padding_mask)

        if extras:
            return (hidden_states, *extras)
        return hidden_states


def notes_to_token_ids(notes, tokenizer):
    # aria's tokenizer only reads files so write a temp midi
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    offset = notes[0][0]

    for note in notes:
        start, end, pitch, velocity = note_fields(note)
        inst.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=int(pitch), start=start - offset, end=end - offset
        ))

    midi.instruments.append(inst)

    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as tmp:
        tmp_path = tmp.name
    midi.write(tmp_path)

    try:
        res = tokenizer.encode_from_file(tmp_path, return_tensors="pt")
    finally:
        os.remove(tmp_path)

    return res.input_ids  # (1, num_tokens)


def motif_notes_to_vector(motif_notes, tokenizer, model):
    ids = notes_to_token_ids(motif_notes, tokenizer)

    # mean pool the token embeddings (table is frozen)
    with torch.no_grad():
        embeds = model.model.tok_embeddings(ids)  # (1, num_tokens, 1536)
        pooled = embeds.mean(dim=1)

    return pooled.squeeze(0)  # (1536,)


def get_motif_vecs(example, tokenizer, model, max_motifs=3):
    vecs = []
    lens = example.get("motif_lens", {})  # claude's motifs aren't all 7 notes

    for key, positions in list(example["motifs"].items())[:max_motifs]:
        pos = positions[0]
        motif_notes = example["context"][pos:pos + lens.get(key, MOTIF_LEN)]

        if len(motif_notes) < 2:
            continue

        vecs.append(motif_notes_to_vector(motif_notes, tokenizer, model))

    return vecs


def lm_loss(logits, input_ids, ignore_index=-100):
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        ignore_index=ignore_index,
    )


def train_one_epoch(model, patched_block, tokenizer, precomputed_dir, optimizer, max_examples=None, checkpoint_every=200, checkpoint_path="checkpoint.pt", use_wandb=False, wandb_project="motif-memory", wandb_run_name=None, accumulation_steps=1):
    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    model.train()
    total_loss = 0.0
    count = 0
    start_time = time.time()
    optimizer.zero_grad()

    for entry in piece_index:
        if max_examples is not None and count >= max_examples:
            break

        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            examples = json.load(f)

        for example in examples:
            if max_examples is not None and count >= max_examples:
                break

            motif_vecs = get_motif_vecs(example, tokenizer, model)
            if not motif_vecs:
                continue

            patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0)
            patched_block.motif_padding_mask = None

            full_notes = example["context"] + example["target"]
            if len(full_notes) < 2:
                continue

            try:
                input_ids = notes_to_token_ids(full_notes, tokenizer)
            except Exception:
                continue

            input_ids = input_ids[:, :MAX_TOKENS]
            if input_ids.shape[1] < 2:
                continue

            logits = model(input_ids)[0]
            loss = lm_loss(logits, input_ids)
            (loss / accumulation_steps).backward()

            count += 1

            if count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            gate = torch.sigmoid(patched_block.motif_attn.gate).item()

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "gate": gate,
                    "avg_loss_so_far": total_loss / count,
                    "example": count,
                })

            if count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"example {count}: loss = {loss.item():.4f}, gate = {gate:.4f}, "
                      f"{elapsed / count:.2f}s/example, elapsed {elapsed/60:.1f}min")

            if count % checkpoint_every == 0:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  checkpoint saved at example {count}")

    # leftover grads from a partial accumulation
    if count % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\ndone: {count} examples, avg loss = {total_loss / max(count, 1):.4f}")
    print(f"saved to {checkpoint_path}")

    if use_wandb:
        wandb.finish()


def generate_continuation(model, tokenizer, patched_block, midi_path, num_tokens=100, use_motifs=True, motif_vector=None):
    prompt = tokenizer.encode_from_file(midi_path, return_tensors="pt")
    input_ids = prompt.input_ids[..., :100]  # first 100 tokens as the seed

    if use_motifs and motif_vector is not None:
        patched_block.motif_vectors = motif_vector.unsqueeze(0).unsqueeze(0)
    else:
        patched_block.motif_vectors = None

    model.eval()
    with torch.no_grad():
        return model.generate(input_ids, max_new_tokens=num_tokens, do_sample=True, temperature=0.9)


def run_ablation_sweep(base_model_name, tokenizer, precomputed_dir, layer_idx=14, examples_per_run=800):
    baseline = {"lr": 1e-5, "dropout": 0.0, "accum": 1}

    runs = [
        {**baseline, "name": "baseline"},
        {**baseline, "lr": 1e-6, "name": "lr_1e-6"},
        {**baseline, "lr": 1e-4, "name": "lr_1e-4"},
        {**baseline, "dropout": 0.1, "name": "dropout_0.1"},
        {**baseline, "dropout": 0.2, "name": "dropout_0.2"},
        {**baseline, "accum": 4, "name": "accum_4"},
        {**baseline, "accum": 16, "name": "accum_16"},
    ]

    done = []
    for cfg in runs:
        print(f"\nrun: {cfg['name']} (lr={cfg['lr']}, dropout={cfg['dropout']}, accum={cfg['accum']})")

        # fresh model every run so they don't affect each other
        model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

        for p in model.model.tok_embeddings.parameters():
            p.requires_grad = False

        layers = model.model.encode_layers
        layers[layer_idx] = BlockWithMotifAttention(layers[layer_idx], model.config.hidden_size, dropout=cfg["dropout"])

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])

        train_one_epoch(
            model, layers[layer_idx], tokenizer, precomputed_dir, optimizer,
            max_examples=examples_per_run, use_wandb=True, wandb_run_name=cfg["name"],
            accumulation_steps=cfg["accum"], checkpoint_path=f"checkpoint_{cfg['name']}.pt",
        )

        done.append(cfg["name"])

    print(f"\nall {len(done)} runs done: {done}")
    return done


def load_maestro_splits(csv_path):
    splits = {}

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # csv has "2018/MIDI-....midi", the piece ids are "2018_MIDI-....midi"
            piece_id = row["midi_filename"].replace("/", "_")
            splits[piece_id] = row["split"]

    return splits


def group_index_by_split(precomputed_dir, splits):
    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    grouped = {"train": [], "validation": [], "test": []}
    unmatched = 0

    for entry in piece_index:
        split = splits.get(entry["piece_id"])
        if split in grouped:
            grouped[split].append(entry)
        else:
            unmatched += 1

    print(f"grouped pieces: train={len(grouped['train'])}, "
          f"validation={len(grouped['validation'])}, test={len(grouped['test'])}, "
          f"unmatched={unmatched}")

    return grouped


def train_on_split_only(model, patched_block, tokenizer, optimizer, precomputed_dir, splits, target_split="train", max_examples=300):
    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    model.train()
    count = 0
    total_loss = 0.0

    for entry in piece_index:
        if count >= max_examples:
            break
        if splits.get(entry["piece_id"]) != target_split:
            continue

        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            examples = json.load(f)

        for example in examples:
            if count >= max_examples:
                break

            motif_vecs = get_motif_vecs(example, tokenizer, model)
            if not motif_vecs:
                continue
            patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0)

            full_notes = example["context"] + example["target"]
            try:
                input_ids = notes_to_token_ids(full_notes, tokenizer)[:, :MAX_TOKENS]
            except Exception:
                continue
            if input_ids.shape[1] < 2:
                continue

            loss = lm_loss(model(input_ids)[0], input_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if count % 20 == 0:
                print(f"train example {count}: loss={loss.item():.4f}")

    print(f"trained on {count} examples from '{target_split}', avg loss={total_loss/max(count,1):.4f}")


def eval_on_split_only(model, patched_block, tokenizer, precomputed_dir, splits, target_split="validation", max_examples=100, use_motifs=True):
    patched_block.motif_padding_mask = None  # can be left over from batched training

    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    model.eval()
    total_loss = 0.0
    count = 0

    for entry in piece_index:
        if count >= max_examples:
            break
        if splits.get(entry["piece_id"]) != target_split:
            continue

        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            examples = json.load(f)

        for example in examples:
            if count >= max_examples:
                break

            if use_motifs:
                motif_vecs = get_motif_vecs(example, tokenizer, model)
                if not motif_vecs:
                    continue
                patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0)
            else:
                patched_block.motif_vectors = None

            full_notes = example["context"] + example["target"]
            try:
                input_ids = notes_to_token_ids(full_notes, tokenizer)[:, :MAX_TOKENS]
            except Exception:
                continue
            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                loss = lm_loss(model(input_ids)[0], input_ids)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    return count, avg_loss, math.exp(avg_loss)


def build_batch(examples, tokenizer, model, pad_token_id=PAD_ID, max_motifs=3, max_seq_len=MAX_TOKENS):
    all_ids = []
    all_motifs = []

    for example in examples:
        motif_vecs = get_motif_vecs(example, tokenizer, model, max_motifs)
        full_notes = example["context"] + example["target"]
        try:
            ids = notes_to_token_ids(full_notes, tokenizer)[0, :max_seq_len]
        except Exception:
            continue

        if ids.shape[0] < 2 or not motif_vecs:
            continue

        all_ids.append(ids)
        all_motifs.append(torch.stack(motif_vecs))

    if not all_ids:
        return None

    n = len(all_ids)
    max_len = max(t.shape[0] for t in all_ids)

    padded_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)

    for i, t in enumerate(all_ids):
        padded_ids[i, :t.shape[0]] = t
        attention_mask[i, :t.shape[0]] = 1

    # same thing for the motifs
    max_motifs_in_batch = max(m.shape[0] for m in all_motifs)
    hidden_dim = all_motifs[0].shape[1]
    padded_motifs = torch.zeros((n, max_motifs_in_batch, hidden_dim))
    motif_padding_mask = torch.ones((n, max_motifs_in_batch), dtype=torch.bool)

    for i, m in enumerate(all_motifs):
        padded_motifs[i, :m.shape[0]] = m
        motif_padding_mask[i, :m.shape[0]] = False  # True = padding

    return padded_ids, attention_mask, padded_motifs, motif_padding_mask


def train_with_epochs(model, patched_block, tokenizer, precomputed_dir, optimizer, splits, num_epochs=3, checkpoint_path="checkpoint.pt", use_wandb=False, wandb_project="motif-memory", wandb_run_name=None, batch_size=4):
    grouped = group_index_by_split(precomputed_dir, splits)

    # load every train example up front so batches can mix pieces
    train_examples = []
    for entry in grouped["train"]:
        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            train_examples.extend(json.load(f))

    total_steps = (len(train_examples) // batch_size) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    step = 0
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for i in range(0, len(train_examples), batch_size):
            batch = build_batch(train_examples[i:i + batch_size], tokenizer, model)
            if batch is None:
                continue
            input_ids, attention_mask, motif_vectors, motif_padding_mask = batch

            patched_block.motif_vectors = motif_vectors
            patched_block.motif_padding_mask = motif_padding_mask

            logits = model(input_ids, attention_mask=attention_mask)[0]
            loss = lm_loss(logits, input_ids, ignore_index=PAD_ID)

            optimizer.zero_grad()
            loss.backward()
            # max_norm=inf so this just measures the norm, doesn't clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=float("inf")
            )
            optimizer.step()
            scheduler.step()

            step += 1
            epoch_loss += loss.item()
            epoch_count += 1

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "gate": torch.sigmoid(patched_block.motif_attn.gate).item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "global_grad_norm": grad_norm.item(),
                    "epoch": epoch,
                    "step": step,
                })

        train_loss = epoch_loss / max(epoch_count, 1)
        print(f"epoch {epoch} done: {epoch_count} batches ({epoch_count * batch_size} examples, batch_size={batch_size}), avg loss = {train_loss:.4f}")

        val_count, val_loss, val_ppl = eval_on_split_only(
            model, patched_block, tokenizer, precomputed_dir, splits,
            target_split="validation", max_examples=30, use_motifs=True
        )
        print(f"epoch {epoch} validation: {val_count} examples, loss = {val_loss:.4f}, perplexity = {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if use_wandb:
            wandb.log({"epoch": epoch, "step": step, "val_loss": val_loss, "val_perplexity": val_ppl})

        # one file per epoch, used to overwrite the same file so only the last epoch survived
        path = checkpoint_path.replace(".pt", f"_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        print(f"  saved to {path}")

    print(f"\nbest epoch by val loss: epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    if use_wandb:
        wandb.finish()


def select_fixed_examples(precomputed_dir, splits, n=10, seed=42):
    # seeded so every run looks at the same examples
    import random

    rng = random.Random(seed)

    grouped = group_index_by_split(precomputed_dir, splits)
    fixed = {}

    for split in ["train", "validation", "test"]:
        pieces = rng.sample(grouped[split], min(n, len(grouped[split])))

        chosen = []

        for entry in pieces:
            with open(os.path.join(precomputed_dir, entry["file"])) as f:
                piece_examples = json.load(f)

            if piece_examples:
                chosen.append({"piece_id": entry["piece_id"], "example": piece_examples[0]})

        fixed[split] = chosen
        print(f"{split}: picked {len(chosen)} examples")

    return fixed


def plot_piano_roll(notes, title, save_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    for note in notes:
        ax.plot([note[0], note[1]], [note[2], note[2]], linewidth=4, color="steelblue")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"saved {save_path}")


def notes_to_midi(notes):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    for note in notes:
        start, end, pitch, velocity = note_fields(note)
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=start, end=end))

    midi.instruments.append(inst)
    return midi


def visualize_example(example_entry, model, tokenizer, patched_block, output_dir, label):
    """piano roll + wav of the ground truth and a generated continuation"""
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)
    example = example_entry["example"]
    piece_id = example_entry["piece_id"]
    out = lambda name: os.path.join(output_dir, f"{label}_{piece_id}_{name}")

    # melody context + polyphonic target
    truth = example["context"] + example["target"]
    plot_piano_roll(truth, f"{label} ground truth: {piece_id}", out("ground_truth.png"))

    motif_vecs = get_motif_vecs(example, tokenizer, model)

    # capped like in training, some pieces blow past aria's 8192 limit
    try:
        context_ids = notes_to_token_ids(example["context"], tokenizer)[:, :MAX_TOKENS]
    except Exception as e:
        print(f"skipping {piece_id}: tokenizing failed ({e})")
        return

    if context_ids.shape[1] < 2:
        print(f"skipping {piece_id}: context too short")
        return

    patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0) if motif_vecs else None
    patched_block.motif_padding_mask = None

    model.eval()
    with torch.no_grad():
        generated = model.generate(context_ids, max_new_tokens=100, do_sample=True, temperature=0.9)

    # decode gives a mido file, go through a temp file to get pretty_midi
    mido_midi = tokenizer.decode(generated[0].tolist()).to_midi()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    mido_midi.save(tmp_path)

    try:
        gen_midi = pretty_midi.PrettyMIDI(tmp_path)
    finally:
        os.remove(tmp_path)

    gen_notes = [[n.start, n.end, n.pitch] for inst in gen_midi.instruments for n in inst.notes]
    plot_piano_roll(gen_notes, f"{label} generated: {piece_id}", out("generated.png"))

    # synthesize() is just sine waves but doesn't need a soundfont
    sf.write(out("ground_truth.wav"), notes_to_midi(truth).synthesize(), 44100)
    sf.write(out("generated.wav"), gen_midi.synthesize(), 44100)
    print(f"saved audio for {piece_id}")


def generate_with_motif_switching(model, tokenizer, patched_block, seed_notes, motif_sequence, tokens_per_segment=50):
    """generate in segments, segment i uses motif_sequence[i]. to hear if the output reacts"""
    ids = notes_to_token_ids(seed_notes, tokenizer)[:, :MAX_TOKENS]

    model.eval()
    for i, motif_notes in enumerate(motif_sequence):
        vec = motif_notes_to_vector(motif_notes, tokenizer, model)
        patched_block.motif_vectors = vec.unsqueeze(0).unsqueeze(0)
        patched_block.motif_padding_mask = None

        with torch.no_grad():
            ids = model.generate(ids, max_new_tokens=tokens_per_segment, do_sample=True, temperature=0.9)
        print(f"segment {i+1}: {tokens_per_segment} tokens with motif {i+1}")

    return ids


def test_gate_forcing(model, patched_block, tokenizer, seed_notes, motif_notes, gate_override):
    # gate_override is pre sigmoid, -10 is basically off and +10 basically on
    with torch.no_grad():
        patched_block.motif_attn.gate.fill_(gate_override)

    vec = motif_notes_to_vector(motif_notes, tokenizer, model)
    patched_block.motif_vectors = vec.unsqueeze(0).unsqueeze(0)
    patched_block.motif_padding_mask = None

    ids = notes_to_token_ids(seed_notes, tokenizer)[:, :MAX_TOKENS]

    model.eval()
    with torch.no_grad():
        return model(ids)[0]


def build_batch_token_concatenated(examples, pad_token_id=PAD_ID, max_seq_len=1024):
    # labels are -100 before the target so the loss only counts target tokens
    seqs = []
    target_starts = []

    for ex in examples:
        if ex is None:
            continue
        seq = ex["input_ids"][:max_seq_len]

        seqs.append(seq)
        target_starts.append(min(ex["target_start_index"], seq.shape[0]))

    if not seqs:
        return None

    max_len = max(s.shape[0] for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)

    for i, seq in enumerate(seqs):
        input_ids[i, :seq.shape[0]] = seq
        attention_mask[i, :seq.shape[0]] = 1
        labels[i, target_starts[i]:seq.shape[0]] = seq[target_starts[i]:]

    return input_ids, attention_mask, labels


def train_token_concatenated(model, tokenizer, pieces, optimizer, num_epochs=1, batch_size=4, use_wandb=False, wandb_project="motif-memory", wandb_run_name=None, checkpoint_path="checkpoint_tokens.pt"):
    """plain aria, no patched block. pieces is a list of (raw_notes, melody, motifs), no precompute for this yet"""
    from doingstuff import build_token_concatenated_example

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    step = 0
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_count = 0

        batch_examples = []
        for raw_notes, melody, motifs in pieces:
            ex = build_token_concatenated_example(raw_notes, melody, motifs, tokenizer)
            if ex is None:
                continue
            batch_examples.append(ex)

            if len(batch_examples) < batch_size:
                continue

            batch = build_batch_token_concatenated(batch_examples)
            batch_examples = []

            if batch is None:
                continue
            input_ids, attention_mask, labels = batch

            logits = model(input_ids, attention_mask=attention_mask)[0]
            loss = lm_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item()
            epoch_count += 1

            if use_wandb:
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})

            if step % 10 == 0:
                print(f"step {step}: loss = {loss.item():.4f}")

        print(f"epoch {epoch} done: {epoch_count} batches, avg loss = {epoch_loss / max(epoch_count, 1):.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved to {checkpoint_path}")

    if use_wandb:
        wandb.finish()
