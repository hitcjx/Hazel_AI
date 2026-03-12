#!/bin/bash

# Hazel AI 启动脚本
# 前后端都在同一台服务器上运行

echo "========================================"
echo "🚀 启动 Hazel AI 系统"
echo "========================================"

# 停止旧进程
echo "🛑 停止旧进程..."
pkill -f "next dev" 2>/dev/null || true
pkill -f "uvicorn.*main:app" 2>/dev/null || true
pkill -f "python.*backend/main.py" 2>/dev/null || true
sleep 2

# 激活conda环境
echo "📦 激活conda环境..."
source /home/dazzle/miniconda3/etc/profile.d/conda.sh
conda activate hazel_ai

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ 错误：conda环境激活失败"
    exit 1
fi
echo "✅ Conda环境: $CONDA_DEFAULT_ENV"

# 检查Python路径
PYTHON_PATH=$(which python)
echo "🐍 Python路径: $PYTHON_PATH"

# 启动后端（FastAPI）
echo "🔧 启动后端服务（FastAPI）..."
echo "⏳ 提示：首次启动可能需要1-2分钟加载模型，请耐心等待..."
cd /home/dazzle/Hazel_AI

# 使用conda环境中的Python直接运行main.py
nohup python backend/main.py > /tmp/hazel_backend.log 2>&1 &
BACKEND_PID=$!
echo "📝 后端PID: $BACKEND_PID"

# 等待后端启动（延长超时时间到120秒，适应首次加载模型）
echo "⏳ 等待后端启动（最长120秒）..."
BACKEND_STARTED=false
for i in {1..120}; do
    sleep 1
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 后端已启动：http://localhost:8000 (耗时 ${i} 秒)"
        BACKEND_STARTED=true
        break
    fi
    # 每10秒显示一次提示
    if [ $((i % 10)) -eq 0 ]; then
        echo "⏳ 还在等待后端启动... (${i}/120秒)"
        # 显示最后几行日志
        echo "📋 最新日志："
        tail -3 /tmp/hazel_backend.log | sed 's/^/   /'
    fi
done

if [ "$BACKEND_STARTED" = false ]; then
    echo "❌ 错误：后端启动超时（120秒）"
    echo "📋 完整后端日志："
    cat /tmp/hazel_backend.log | sed 's/^/   /'
    echo ""
    echo "💡 可能的原因："
    echo "   1. 端口8000被占用"
    echo "   2. 模型加载失败"
    echo "   3. Python依赖缺失"
    echo ""
    echo "🔍 检查端口占用："
    netstat -tuln 2>/dev/null | grep ":8000" || echo "   端口8000未被占用"
    exit 1
fi

# 启动前端（Next.js）
echo "🎨 启动前端服务（Next.js）..."
cd /home/dazzle/Hazel_AI/frontend

# 检查node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 首次启动，安装依赖..."
    npm install
fi

nohup npm run dev > /tmp/hazel_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "📝 前端PID: $FRONTEND_PID"

# 等待前端启动
echo "⏳ 等待前端启动..."
FRONTEND_STARTED=false
for i in {1..30}; do
    sleep 1
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ 前端已启动：http://localhost:3000 (耗时 ${i} 秒)"
        FRONTEND_STARTED=true
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "⏳ 还在等待前端启动... (${i}/30秒)"
    fi
done

if [ "$FRONTEND_STARTED" = false ]; then
    echo "⚠️  警告：前端启动超时，请手动检查"
    echo "📋 前端日志："
    tail -20 /tmp/hazel_frontend.log | sed 's/^/   /'
fi

echo ""
echo "========================================"
echo "✅ Hazel AI 系统启动完成！"
echo "========================================"
echo "📝 前端地址：http://localhost:3000"
echo "🔧 后端API：http://localhost:8000"
echo "📚 API文档：http://localhost:8000/docs"
echo "========================================"
echo "📋 进程信息："
echo "   - 后端PID: $BACKEND_PID"
echo "   - 前端PID: $FRONTEND_PID"
echo "========================================"
echo "📋 查看日志："
echo "   - 后端: tail -f /tmp/hazel_backend.log"
echo "   - 前端: tail -f /tmp/hazel_frontend.log"
echo "========================================"
echo "💡 停止服务："
echo "   - kill $BACKEND_PID $FRONTEND_PID"
echo "   - 或运行: pkill -f 'next dev'; pkill -f 'uvicorn'"
echo "========================================"
echo ""
