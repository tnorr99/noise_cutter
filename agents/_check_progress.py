import os, time
from datetime import datetime, timedelta

waterloo_dir = r'C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\data\triplet_dataset\waterloo'

completed = sorted([
    (f, os.path.getmtime(os.path.join(waterloo_dir, f)))
    for f in os.listdir(waterloo_dir)
    if os.path.isfile(os.path.join(waterloo_dir, f, 'upscaled.mp4'))
], key=lambda x: x[1])

in_progress = [
    f for f in os.listdir(waterloo_dir)
    if os.path.isfile(os.path.join(waterloo_dir, f, 'original.mp4'))
    and not os.path.isfile(os.path.join(waterloo_dir, f, 'upscaled.mp4'))
]

intervals = [(completed[i+1][1] - completed[i][1]) / 60
             for i in range(max(0, len(completed)-6), len(completed)-1)]
avg = sum(intervals) / len(intervals)

print(f"Completed : {len(completed)} / 80")
print(f"In progress: {in_progress[0] if in_progress else 'none'}")
print(f"\nLast 5 completion times:")
for name, mtime in completed[-5:]:
    print(f"  {name}  {datetime.fromtimestamp(mtime).strftime('%b %d %H:%M')}")
print(f"\nAvg per video : {avg:.0f} min")

if in_progress:
    ip_mtime = os.path.getmtime(os.path.join(waterloo_dir, in_progress[0]))
    elapsed = (time.time() - ip_mtime) / 60
    remaining_current = max(0, avg - elapsed)
    queued = 80 - len(completed) - 1
    total_min = remaining_current + queued * avg
    eta = datetime.now() + timedelta(minutes=total_min)
    print(f"Current elapsed  : {elapsed:.0f} min  (~{remaining_current:.0f} min left)")
    print(f"Videos remaining : {queued} after current")
    print(f"Total time left  : {total_min:.0f} min  ({total_min/60:.1f} hrs)")
    print(f"ETA              : {eta.strftime('%A %b %d at %H:%M')}")
