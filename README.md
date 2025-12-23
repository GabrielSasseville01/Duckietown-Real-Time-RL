# Duckietown Real-Time Reinforcement Learning

Welcome! This repository contains implementations of Reinforcement Learning algorithms (SAC, PPO, REINFORCE) for autonomous vehicle control in the Duckietown environment, with support for real-time training and simulation with computation delays.

## 📚 Repository Branches

This repository uses two branches for different purposes:

### `main` Branch (Current Branch)
- **Purpose**: Contains the blog post and high-level project overview
- **Content**: This README and blog-related materials
- **Audience**: Visitors who want to learn about the project or read the associated blog post

### `code` Branch
- **Purpose**: Contains the complete implementation code and detailed documentation
- **Content**: 
  - Full source code for RL algorithms (SAC, PPO, REINFORCE)
  - Complete installation and usage instructions
  - Project structure documentation
  - Experiment scripts and analysis tools
  - Comprehensive README with all technical details
- **Audience**: Developers and researchers who want to run experiments, understand the codebase, or reproduce results

## 🚀 Getting Started

### To Access the Code and Documentation

If you're looking to:
- Run experiments
- Understand the code structure
- Install and use the RL algorithms
- View detailed documentation

**Please switch to the `code` branch:**

```bash
git checkout code
```

Then read the detailed README in that branch, which includes:
- Complete installation instructions
- Training and evaluation guides
- Project structure explanation (dt-duckiematrix, duckietown-sdk, gym-duckiematrix)
- Hyperparameter tuning and delay experiments
- Troubleshooting guide
- And much more!

### Quick Branch Switch Commands

```bash
# Switch to the code branch
git checkout code

# Switch back to main branch
git checkout main

# View all available branches
git branch -a
```

## 📖 What's in the Code Branch?

The `code` branch contains:

- **RL Algorithms**: Full implementations of SAC, PPO, and REINFORCE
- **Training Scripts**: Ready-to-use training and evaluation scripts
- **Experiment Tools**: Delay experiments, hyperparameter tuning, and analysis utilities
- **Documentation**: Comprehensive guides for installation, usage, and project structure
- **Project Components**:
  - `dt-duckiematrix/`: Duckietown Matrix Engine (simulation core)
  - `duckietown-sdk/`: Duckietown SDK (Python API)
  - `gym-duckiematrix/`: Gymnasium wrapper & RL implementation

## 🔗 Quick Links

- **Blog Post**: [Link to blog post - if available]
- **Code Branch README**: Switch to `code` branch and see `README.md`
- **Repository**: [GitHub repository URL]

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{duckietown-realtime-rl,
  title={Duckietown Real-Time Reinforcement Learning},
  author={Gabriel Sasseville, Guillaume Gagné-Labelle, Nicolas Bosteels},
  year={2025},
  url={https://github.com/GabrielSasseville01/Duckietown-Real-Time-RL}
}
```

## 📄 License

Common BY-SA

---

**Next Step**: Run `git checkout code` to access the full implementation and documentation!
