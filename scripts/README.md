# 🔧 Project Scripts

Essential setup and utility scripts for the Football Analysis project.

## 📊 Data Scripts

### **`extract_football_data.py`**
Extract football videos from your 298GB SportsNet dataset to C drive.
```bash
python extract_football_data.py --source "D:/Sportspose" --output "C:/FootballData/dataset" --sample_size 50
```

### **`explore_dataset.py`**
Analyze and explore your dataset structure.
```bash
python explore_dataset.py
```

## 🏗️ Setup Scripts

### **`setup_football_project.py`**
Create complete football app project structure.
```bash
python setup_football_project.py
```

## 🎯 Usage Examples

### **Complete Setup Workflow:**
```bash
# 1. Extract football data
python scripts/extract_football_data.py --source "D:/Sportspose" --output "C:/FootballData/dataset"

# 2. Create project structure
python scripts/setup_football_project.py

# 3. Explore what you have
python scripts/explore_dataset.py
```

### **Quick Data Extraction:**
```bash
# Small sample for testing
python scripts/extract_football_data.py --source "D:/Sportspose" --output "C:/FootballData/dataset" --sample_size 20

# Larger sample for training
python scripts/extract_football_data.py --source "D:/Sportspose" --output "C:/FootballData/dataset" --sample_size 100
```

---

**All scripts are designed to work with your C drive data storage strategy!** 💾