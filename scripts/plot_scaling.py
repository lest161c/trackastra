import matplotlib.pyplot as plt
import csv
import argparse

def plot_scaling_results(csv_file, output_file="scaling_achievements.png"):
    nodes_def, time_def, mem_def = [], [], []
    nodes_knn, time_knn, mem_knn = [], [], []

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["Tokens"])
            if row["Mode"] == "Default":
                nodes_def.append(n)
                if row["Time_ms"] != "OOM":
                    time_def.append(float(row["Time_ms"]))
                    mem_def.append(float(row["VRAM_MB"]))
                else:
                    time_def.append(None)
                    mem_def.append(None)
            elif row["Mode"] == "KNN":
                nodes_knn.append(n)
                if row["Time_ms"] != "OOM":
                    time_knn.append(float(row["Time_ms"]))
                    mem_knn.append(float(row["VRAM_MB"]))
                else:
                    time_knn.append(None)
                    mem_knn.append(None)

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Trackastra Attention Scaling Achievements: Default vs. KNN', fontsize=16, fontweight='bold')

    # Time Plot
    ax1.plot(nodes_def, time_def, 'o-', color='#e74c3c', label='Default Attention (Full N²)', linewidth=2.5, markersize=8)
    ax1.plot(nodes_knn, time_knn, 's-', color='#3498db', label='KNN Attention (O(N·K))', linewidth=2.5, markersize=8)
    ax1.set_title('Execution Time per Step (Forward + Backward)', fontsize=12)
    ax1.set_xlabel('Number of Tokens (Cells in Window)', fontsize=11)
    ax1.set_ylabel('Execution Time (ms)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=11)

    # Memory Plot
    ax2.plot(nodes_def, mem_def, 'o-', color='#e74c3c', label='Default Attention (Full N²)', linewidth=2.5, markersize=8)
    ax2.plot(nodes_knn, mem_knn, 's-', color='#3498db', label='KNN Attention (O(N·K))', linewidth=2.5, markersize=8)
    ax2.set_title('Peak VRAM Allocation per Step', fontsize=12)
    ax2.set_xlabel('Number of Tokens (Cells in Window)', fontsize=11)
    ax2.set_ylabel('Peak VRAM (MB)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved successfully to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="scaling_results.csv")
    parser.add_argument("--output", type=str, default="scaling_achievements.png")
    args = parser.parse_args()
    
    plot_scaling_results(args.input, args.output)
