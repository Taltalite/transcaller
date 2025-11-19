import re
import matplotlib.pyplot as plt
import os

def set_chinese_font():
    """
    Attempts to set a Chinese font for matplotlib to display characters correctly.
    """
    try:
        # 'SimHei' is a common font on Windows
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # To display minus sign correctly
        
        # Test if it works
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '测试', ha='center', va='center')
        plt.close(fig)
        print("Font 'SimHei' loaded successfully for Chinese characters.")
    except Exception:
        try:
            # 'WenQuanYi Zen Hei' is common on Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
            plt.rcParams['axes.unicode_minus'] = False
            # Test
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, '测试', ha='center', va='center')
            plt.close(fig)
            print("Font 'WenQuanYi Zen Hei' loaded successfully for Chinese characters.")
        except Exception:
            print("Warning: Could not set a Chinese font (SimHei, WenQuanYi Zen Hei).")
            print("Plot labels may not display Chinese characters correctly.")
            print("Please install a compatible font (e.g., 'SimHei') and ensure matplotlib can find it.")

def parse_and_plot(log_file_path):
    """
    Parses a training log file to extract epoch, train loss, and validation loss,
    then plots the results.
    """
    
    # Set up matplotlib to support Chinese characters
    set_chinese_font()

    # Define regex patterns
    # Pattern for "--- Epoch X/Y ---"
    epoch_pattern = re.compile(r"--- Epoch (\d+)/\d+ ---")
    # Pattern for "[总结] 训练损失: X.XXXX | 验证损失: Y.YYYY"
    loss_pattern = re.compile(r"\[总结\] 训练损失: (\d+\.\d+) \| 验证损失: (\d+\.\d+)")

    # Lists to store extracted data
    epoch_numbers = []
    train_losses = []
    val_losses = []

    current_epoch = 0  # To track the current epoch being processed

    # --- Demo File Creation ---
    # Check if the log file exists. If not, create a dummy one for demonstration.
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found at '{log_file_path}'.")
        print("Creating a dummy 'train_log_demo.log' for demonstration.")
        log_file_path = 'train_log_demo.log'
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(
                "--- Epoch 1/30 ---\n"
                "Epoch 1 完成. 耗时: 3000s\n"
                " [总结] 训练损失: 0.5000 | 验证损失: 0.4500\n"
                "(新最佳模型! ...)\n"
                "--- Epoch 2/30 ---\n"
                "Epoch 2 完成. 耗时: 3001s\n"
                " [总结] 训练损失: 0.4000 | 验证损失: 0.3500\n"
                "(新最佳模型! ...)\n"
                "--- Epoch 3/30 ---\n"
                "Epoch 3 完成. 耗时: 3002s\n"
                " [总结] 训练损失: 0.3000 | 验证损失: 0.3200\n"
                "--- Epoch 4/30 ---\n"
                "Epoch 4 完成. 耗时: 3003s\n"
                " [总结] 训练损失: 0.2500 | 验证损失: 0.2800\n"
                "(新最佳模型! ...)\n"
            )
    # --- End of Demo File Creation ---
    
    print(f"Parsing log file: {log_file_path}")

    # Read and parse the log file
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Check for epoch start
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    continue  # Move to the next line, as this line doesn't have loss info

                # Check for loss summary line
                loss_match = loss_pattern.search(line)
                
                # We must have found an epoch number first (current_epoch > 0)
                if loss_match and current_epoch > 0:
                    train_loss = float(loss_match.group(1))
                    val_loss = float(loss_match.group(2))

                    # Store the data
                    epoch_numbers.append(current_epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    # Optional: Print extracted data
                    print(f"Epoch {current_epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")

    except FileNotFoundError:
        print(f"Error: File not found at '{log_file_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Check if we found any data
    if not epoch_numbers:
        print("No loss data was successfully extracted.")
        print("Please check the log file format and the regex patterns in the script.")
        return

    # Plot the data
    plt.figure(figsize=(12, 7))
    plt.plot(epoch_numbers, train_losses, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epoch_numbers, val_losses, label='Validation Loss', marker='s', linestyle='--')

    # Add plot titles and labels (in Chinese)
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # Show the legend
    plt.grid(True)  # Add a grid for readability

    # Set integer ticks for x-axis if there are not too many epochs
    if len(epoch_numbers) > 0:
        max_epoch = max(epoch_numbers)
        # Only show integer ticks if total epochs are manageable
        if max_epoch <= 50:
            all_epochs = range(min(epoch_numbers), max_epoch + 1)
            # Only label every 1, 2, or 5 ticks to avoid clutter
            step = 1
            if max_epoch > 20:
                step = 2
            if max_epoch > 40:
                step = 5
            plt.xticks(list(all_epochs)[::step])

    # Save the plot to a file
    plot_filename = 'train_transcaller_light_251111_1_loss_curve.png'
    try:
        plt.savefig(plot_filename)
        print(f"Plot successfully saved as '{plot_filename}'")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this to the actual path of your log file
    log_file = '/home/lijy/workspace/my_basecaller/train/train_transcaller_light_251111_1.log'
    
    parse_and_plot(log_file)