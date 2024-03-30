import matplotlib.pyplot as plt
from collections import Counter

def plot_metrics(log_files):
    col = ['darkorchid', 'firebrick', 'limegreen']
    cdn_names = ['Akamai', 'AWS', 'Google']  # Names of the CDNs for the bar plot

    episodes = []
    rewards = []
    losses = []
    cdn_hits = []

    for idx, log_file in enumerate(log_files):
        with open(log_file, 'r') as f:
            ep = []
            rew = []
            los = []
            cdn = []
            for line in f:
                par = line.split()
                ep.append(int(par[0]))
                rew.append(float(par[-1]))
                los.append(float(par[-3].replace('tensor(', '').replace(')', '')))
                cdn.append(par[-2])

            episodes.append(ep)
            rewards.append(rew)
            losses.append(los)
            cdn_hits.append(cdn)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(ep, rew, label=f'Model {idx}', color=col[idx])
            plt.xlabel('Number of Training Episodes', fontsize=12)
            plt.ylabel('Rewards', fontsize=12)
            plt.title('Rewards vs. Training Episodes', fontsize=14)
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(ep, los, label=f'Model {idx}', color=col[idx])
            plt.xlabel('Number of Training Episodes', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Loss vs. Training Episodes', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Modified subplot for CDN hit frequency as a bar plot
            plt.subplot(1, 3, 3)
            # Count the occurrences of each CDN name
            cdn_counts = Counter(cdn)
            # Ensure counts for all CDN names are present, even if they are 0
            counts = [cdn_counts.get(name, 0) for name in cdn_names]
            plt.bar(cdn_names, counts, color=col[idx], label=f'Model {idx}')
            plt.xlabel('CDN', fontsize=12)
            plt.ylabel('Hit Count', fontsize=12)
            plt.title('CDN Hit Analysis', fontsize=14)
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    log_files = ['img/log_test']
    plot_metrics(log_files)