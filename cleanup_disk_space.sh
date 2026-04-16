#!/bin/bash
# Safe cleanup script to free up disk space

echo "=" * 80
echo "DISK SPACE CLEANUP"
echo "=" * 80
echo ""

# Show current space
echo "Current disk usage:"
df -h . | tail -1
echo ""

# Show what will be deleted
echo "Files to be deleted:"
echo "  - Old/partial training data files"
echo "  - Old model checkpoints (keeping final models)"
echo "  - Old log files"
echo ""

read -p "Continue with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

echo ""
echo "Cleaning up..."

# Remove old/partial training data
echo "  Removing old training data files..."
rm -f artifacts/training_data/episodic_obs*.npz
rm -f artifacts/training_data/*_old.npz
rm -f artifacts/training_data/*_partial.npz

# Remove old model checkpoints (keep final models)
echo "  Removing old model checkpoints..."
find artifacts/transfer_models -name "*_iter_*.pt" -type f -delete

# Remove old logs
echo "  Removing old logs..."
rm -f artifacts/*.log
rm -f artifacts/logs/*.log

# Show space after cleanup
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Disk usage after cleanup:"
df -h . | tail -1
