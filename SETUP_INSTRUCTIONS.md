# 🚀 Repository Setup Instructions

## Issue: Large Model Files

Your repository contains large AI model files that exceed GitHub's 100MB limit:
- `3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr` (135.36 MB)
- `3dsp_utils/MotionAGFormer/checkpoint/yolov3.weights` (236.52 MB)

## 🔧 Solution Options

### Option 1: Clean Repository (Recommended)
Create a clean repository without the large files:

```bash
# 1. Create a new directory for clean repo
mkdir barca-motion-ai-clean
cd barca-motion-ai-clean

# 2. Initialize new git repo
git init
git branch -M main

# 3. Copy all files EXCEPT the large model files
# (Copy everything from your current project except the checkpoint files)

# 4. Add remote
git remote add origin https://github.com/yourusername/barca-motion-ai.git

# 5. Commit and push
git add .
git commit -m "🏆 Initial commit: Barca Motion AI - FC Barcelona Gen AI Hackathon"
git push -u origin main
```

### Option 2: Use Git LFS (Advanced)
If you need the model files in the repository:

```bash
# 1. Install Git LFS
git lfs install

# 2. Track large files
git lfs track "*.pth"
git lfs track "*.weights"
git lfs track "*.pt"

# 3. Add .gitattributes
git add .gitattributes

# 4. Remove files from history (requires BFG or filter-branch)
# This is complex and may require starting fresh
```

### Option 3: Model Download Script (Best for Hackathon)
Create a setup script that downloads models after cloning:

```bash
# Create setup_models.py script that downloads required models
# Users run this after cloning the repository
```

## 📋 Files to Exclude from Repository

The following large files should NOT be in your GitHub repository:
- All `.pth` and `.pt` model files
- All `.weights` files  
- Large checkpoint directories
- Dataset files (already excluded)
- Temporary processing outputs

## 🎯 Recommended Approach for Hackathon

For the hackathon submission, I recommend **Option 1** (Clean Repository) because:

1. **Fast Setup**: Judges can clone and run immediately
2. **No LFS Complexity**: Simpler for reviewers
3. **Focus on Code**: Emphasizes your implementation over models
4. **Documentation**: Clear instructions for model setup

## 🔄 Next Steps

1. Choose your preferred option above
2. Create the clean repository
3. Update README with model download instructions
4. Test the setup process
5. Submit to hackathon

## 📞 Need Help?

If you need assistance with any of these options, let me know which approach you'd prefer and I can help implement it.