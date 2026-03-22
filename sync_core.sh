#!/bin/bash

# 0. 定义主分支名称和提交消息
MAIN_BRANCH="develop"
COMMIT_MSG=${1:-"chore: update local repository before syncing core $(date +'%Y-%m-%d %H:%M:%S')"}

# 1. 提交当前仓库的改动
git add .
git commit -m "$COMMIT_MSG"
git push origin $MAIN_BRANCH

# 2. 确保在主分支上
echo "🔄 开始同步 Freqtrade 官方核心代码..."
git checkout $MAIN_BRANCH

# 3. 从官方 (upstream) 获取最新改动
git fetch upstream

# 4. 合并官方更新到你的本地分支
# 使用 --no-edit 自动接受默认的 Merge Message
git merge upstream/$MAIN_BRANCH --no-edit

# 5. 如果合并成功，推送到你自己的 Fork (origin)
if [ $? -eq 0 ]; then
    echo "✅ 合并成功，正在同步到你的 GitHub Fork..."
    git push origin $MAIN_BRANCH
else
    echo "❌ 合并出现冲突，请手动解决冲突后再提交！"
    exit 1
fi

# 6. 检查依赖是否发生变化
echo "📦 检查并更新 Conda 环境依赖..."
# 只有在激活了 conda 环境的情况下执行
if [[ "$CONDA_DEFAULT_ENV" == "freqtrade" ]]; then
    pip install -r requirements.txt
    pip install -e .
    echo "🚀 依赖更新完成！"
else
    echo "⚠️ 未检测到已激活的 freqtrade 环境，自动进行环境切换并更新依赖..."
    # 让当前 shell 进程识别 conda 命令
    eval "$(conda shell.bash hook)"
    conda activate freqtrade
    echo "当前环境: $(conda info --envs | grep '*')"
    pip install -r requirements.txt
    pip install -e .
    echo "🚀 依赖更新完成！"
fi

echo "✨ 核心引擎更新流程结束。"

# 7. 远程更新 Oracle 上的仓库
ssh oci "cd quant && sh update_bot.sh"
echo "✅ Oracle 上的仓库已更新！"