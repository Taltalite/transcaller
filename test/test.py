import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import numpy as np

# Import the same custom modules used in training
import sys
sys.path.append('/home/lijy/workspace/')
from my_basecaller.utils.dataset import BasecallingDataset
from my_basecaller.model.model import BasecallerTransformer
from my_basecaller.train.train import custom_collate_fn # We can reuse the collate_fn from the train script

# --- Updated Mappings ---
INT_TO_CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '-'} # Add blank for visualization
BLANK_TOKEN = 4

# --- Updated Decoder with Debug Info ---
def greedy_ctc_decode(log_probs, blank_label=BLANK_TOKEN):
    """
    Decodes and also returns debug information about the raw model output.
    """
    best_path = torch.argmax(log_probs, dim=1)
    
    # --- Debug Info ---
    # Get unique tokens and their counts in the raw path
    unique_tokens, counts = torch.unique(best_path, return_counts=True)
    token_counts = {INT_TO_CHAR.get(tok.item(), 'UNK'): count.item() for tok, count in zip(unique_tokens, counts)}
    
    # --- Standard Decoding Logic ---
    collapsed_path = [p for i, p in enumerate(best_path) if i == 0 or p != best_path[i-1]]
    decoded_indices = [p for p in collapsed_path if p != blank_label]
    decoded_string = "".join([INT_TO_CHAR[p.item()] for p in decoded_indices])
    
    return decoded_string, token_counts

def test(args):
    # ... (Setup code remains the same) ...
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading data and getting the test split...")
    full_dataset = BasecallingDataset(args.data_path)
    test_size = int(0.1 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size - test_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Using Test set of size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)
    model = BasecallerTransformer().to(device)
    print(f"Loading best model from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ctc_loss = nn.CTCLoss(blank=BLANK_TOKEN, reduction='mean', zero_infinity=True)
    test_total_loss = 0
    visualized_count = 0

    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="[Test]")
        for i, batch in enumerate(test_progress_bar):
            # ... (Loss calculation logic remains the same) ...
            signals = batch['signal'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            label_lens = batch['label_len'].to(device, non_blocking=True)
            current_batch_size = signals.shape[0]
            if current_batch_size == 0: continue
            log_probs = model(signals)
            log_probs_perm = log_probs.permute(1, 0, 2)
            input_lengths = torch.full(size=(current_batch_size,), fill_value=log_probs_perm.shape[0], dtype=torch.long).to(device)
            unpadded_labels = [labels[j, :label_lens[j]] for j in range(current_batch_size)]
            labels_cat = torch.cat(unpadded_labels)
            loss = ctc_loss(log_probs_perm, labels_cat, input_lengths, label_lens)
            test_total_loss += loss.item()
            running_avg_loss = test_total_loss / (i + 1)
            test_progress_bar.set_postfix(avg_test_loss=f"{running_avg_loss:.4f}")

            # --- Updated Visualization Logic ---
            if visualized_count < args.visualize_count:
                for j in range(current_batch_size):
                    if visualized_count >= args.visualize_count: break
                    
                    true_indices = labels[j, :label_lens[j]].cpu().tolist()
                    true_sequence = "".join([INT_TO_CHAR[x] for x in true_indices])

                    single_log_probs = log_probs_perm[:, j, :]
                    predicted_sequence, token_counts = greedy_ctc_decode(single_log_probs)

                    print(f"\n--- Visualization Sample {visualized_count + 1} ---")
                    print(f"  Token Counts : {token_counts}")
                    print(f"  Ground Truth : {true_sequence}")
                    print(f"  Predicted    : {predicted_sequence}")
                    
                    visualized_count += 1

    # ... (Final printout remains the same) ...
    avg_test_loss = test_total_loss / len(test_loader)
    print(f"\n--- Test Results ---")
    print(f"Best model from epoch {checkpoint.get('epoch', 'N/A')} with validation loss {checkpoint.get('best_val_loss', 0.0):.4f}")
    print(f"Final Average Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the best basecalling model.")
    # ... (Arguments remain the same) ...
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize_count", type=int, default=5)
    args = parser.parse_args()
    test(args)