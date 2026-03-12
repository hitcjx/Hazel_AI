"""
Hazel AI - FastAPI后端
心理健康评估系统 - API服务
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
import json
import threading
from datetime import datetime

# 添加父目录到路径，以导入现有的Hazel AI模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接从根目录的main.py导入CleanInterface，避免循环导入
import importlib
spec = importlib.util.spec_from_file_location("hazel_main", os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py"))
hazel_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hazel_main)
CleanInterface = hazel_main.CleanInterface

# 导入数据库管理模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import StudentManager, SessionManagerDB, TeacherManager, AssessmentLogger, init_database

# 导入流式气泡分割器
from core.stream_bubble_splitter import StreamBubbleSplitter

# 初始化数据库表
init_database()
print("✅ 数据库表初始化完成")

# =============================================================================
# FastAPI应用初始化
# =============================================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=get_shared_hazel_instance, daemon=True).start()
    yield

app = FastAPI(
    title="Hazel AI API",
    description="心理健康评估系统 - 核心API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # 允许所有可能的localhost访问方式
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",  # IPv6 localhost
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # 暴露所有响应头
)

# =============================================================================
# 数据模型
# =============================================================================
class Message(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = "default"  # 添加session_id

class ChatResponse(BaseModel):
    content: str
    state: Optional[dict] = None  # 包含 stage, risk_level, turn_count 等

# 登录请求模型
class LoginRequest(BaseModel):
    student_id: str
    password: str

# 登录响应模型
class LoginResponse(BaseModel):
    success: bool
    student_id: Optional[str] = None
    name: Optional[str] = None
    session_id: Optional[str] = None
    message: str

# =============================================================================
# 全局状态管理（用于存储会话）
# =============================================================================
# 注意：生产环境应该使用Redis或数据库存储会话
sessions: Dict[str, CleanInterface] = {}

# =============================================================================
# 全局单例模型实例（所有用户共享）
# =============================================================================
_shared_hazel_instance: CleanInterface = None
_instance_lock = threading.Lock()
_is_loading = False

def get_shared_hazel_instance() -> CleanInterface:
    """获取共享的Hazel AI实例（单例模式）"""
    global _shared_hazel_instance, _is_loading
    if _shared_hazel_instance is None:
        with _instance_lock:
            if _shared_hazel_instance is None:
                if _is_loading:
                    # 模型正在加载中
                    raise Exception("系统正在初始化，请稍候...")
                _is_loading = True
                print("🔄 首次加载模型，这可能需要1-2分钟...")
                _shared_hazel_instance = CleanInterface()
                _is_loading = False
                print("✅ 模型加载完成，所有用户将共享此实例")
    return _shared_hazel_instance

def get_or_create_session(session_id: str = "default", student_id: str = None):
    """
    获取或创建会话（不再创建多个CleanInterface实例）

    返回: (CleanInterface实例, session_id)
    """
    import uuid

    # 规范化session_id
    if session_id == "default" or not session_id:
        session_id = str(uuid.uuid4())

    # 在数据库中记录会话（如果还没有）
    if session_id not in sessions:
        sessions[session_id] = {
            "student_id": student_id or "anonymous",
            "created_at": datetime.now().isoformat()
        }
        # 静默处理数据库错误（session可能已存在）
        try:
            SessionManagerDB.create_session(session_id, student_id or "anonymous")
        except Exception as e:
            print(f"⚠️  Session记录已存在或创建失败: {e}")

    # 返回共享的Hazel实例和session_id
    return get_shared_hazel_instance(), session_id

# =============================================================================
# API路由
# =============================================================================
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Hazel AI API服务",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

# =============================================================================
# 调试API
# =============================================================================
class SetStageRequest(BaseModel):
    session_id: str
    stage: str
    turn_count: Optional[int] = None  # 可选：设置轮数

@app.post("/api/debug/set-stage")
async def set_stage(request: SetStageRequest):
    """设置会话阶段（用于测试调试）"""
    try:
        hazel = get_shared_hazel_instance()
        user_session = hazel.get_session(request.session_id)

        # 导入ConsultationStage
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.state_manager import ConsultationStage

        # 设置阶段
        if request.stage == "WARM_UP_SCAN":
            user_session.current_stage = ConsultationStage.WARM_UP_SCAN
        elif request.stage == "EMPATHY_DEEP_DIVE":
            user_session.current_stage = ConsultationStage.EMPATHY_DEEP_DIVE
        elif request.stage == "REFRAMING_SFBT":
            user_session.current_stage = ConsultationStage.REFRAMING_SFBT
        elif request.stage == "CLOSING_EMPOWERMENT":
            user_session.current_stage = ConsultationStage.CLOSING_EMPOWERMENT
        else:
            return {"success": False, "message": f"未知阶段: {request.stage}"}

        # 如果指定了turn_count，也设置
        if request.turn_count is not None:
            user_session.turn_count = request.turn_count

        return {"success": True, "message": f"阶段已设置为: {request.stage}, 轮数: {request.turn_count}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    学生登录API

    Args:
        request: 包含学号的请求

    Returns:
        登录结果，包含学生信息和session_id
    """
    try:
        # 验证学生是否存在
        student = StudentManager.verify_student(request.student_id)

        if not student:
            return LoginResponse(
                success=False,
                message=f"学号 {request.student_id} 不存在，请检查后重试"
            )

        # 验证密码
        import sqlite3
        conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM students WHERE student_id = ?", (request.student_id,))
        row = cursor.fetchone()
        conn.close()

        if not row or row[0] != request.password:
            return LoginResponse(
                success=False,
                message="密码错误，请检查后重试"
            )

        # 创建新会话（首次登录会加载模型，需要1-2分钟）
        hazel, session_id = get_or_create_session("default", student["student_id"])

        # 也在Hazel中创建对应的SessionState
        hazel.get_session(session_id, student["student_id"])

        return LoginResponse(
            success=True,
            student_id=student["student_id"],
            name=student["name"],
            session_id=session_id,
            message=f"欢迎，{student['name']}！"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return LoginResponse(
            success=False,
            message=f"登录失败：{str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天API接口（非流式，兼容旧版）

    Args:
        request: 包含消息历史的请求

    Returns:
        AI回复内容
    """
    try:
        # 获取最后一条用户消息
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 获取共享Hazel实例
        hazel = get_shared_hazel_instance()

        # 获取该用户的独立SessionState
        session_id = request.session_id or "default"

        # 获取session的student_id
        session_info = sessions.get(session_id, {})
        student_id = session_info.get("student_id", session_id)

        user_session = hazel.get_session(session_id, student_id)

        # 调用Hazel AI核心逻辑，传入用户独立的session
        response = hazel.chat_once_with_session(user_message, user_session)

        # 获取当前状态
        # 【调试】打印session状态值
        print(f"\n🐛 [DEBUG] Backend状态构造 (Turn {user_session.turn_count}):")
        print(f"  - user_session.action: '{user_session.action}'")
        print(f"  - user_session.previous_action_status: '{user_session.previous_action_status}'")
        print(f"  - user_session.last_method: '{user_session.last_method}'")
        if user_session.logs:
            print(f"  - logs[-1].sfbt_raw_output: {user_session.logs[-1].sfbt_raw_output[:100] if user_session.logs[-1].sfbt_raw_output else ''}...")

        state = {
            "stage": user_session.current_stage.name if hasattr(user_session.current_stage, 'name') else str(user_session.current_stage),
            "risk_level": user_session.risk_level.name if hasattr(user_session.risk_level, 'name') else str(user_session.risk_level),
            "turn_count": user_session.turn_count,
            "resistance_level": user_session.resistance.level.name if hasattr(user_session.resistance.level, 'name') else str(user_session.resistance.level),
            # SFBT 策略器信息
            "sfbt_method": user_session.last_method or "",
            "sfbt_current_module": user_session.last_current_module or "",
            "sfbt_action": user_session.action or "",
            "sfbt_action_status": user_session.previous_action_status or "",
            "sfbt_plugin": user_session.last_plugin if hasattr(user_session, 'last_plugin') else "",
            "sfbt_score": round(user_session.last_score, 1) if hasattr(user_session, 'last_score') else 0,
            # 获取最后一轮的 SFBT 原始 JSON 输出
            "sfbt_raw": user_session.logs[-1].sfbt_raw_output if user_session.logs else "",
        }

        # 【调试】打印返回的state值
        print(f"  - state['sfbt_action']: '{state['sfbt_action']}'")
        print(f"  - state['sfbt_action_status']: '{state['sfbt_action_status']}'")

        return ChatResponse(content=response, state=state)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_session(request: dict = None):
    """重置会话"""
    # 获取请求中的session_id，如果没有则使用"default"
    session_id = "default"
    if request and "session_id" in request:
        session_id = request["session_id"]

    # 清除Hazel实例中的SessionState
    hazel = get_shared_hazel_instance()
    hazel.clear_session(session_id)

    # 清除后端内存中的会话记录
    if session_id in sessions:
        del sessions[session_id]

    return {"message": f"Session {session_id} reset successfully"}

# =============================================================================
# 教师管理API
# =============================================================================
@app.post("/api/session/save")
async def save_session_data(session_id: str):
    """
    保存会话数据到数据库

    当会话结束时调用，保存：
    1. 对话轮次 (turn_count)
    2. 风险等级 (risk_level)
    3. 评估数据（仅对 MEDIUM/HIGH 风险学生）

    Args:
        session_id: 会话ID

    Returns:
        保存结果
    """
    try:
        import sqlite3
        conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
        cursor = conn.cursor()

        # 查询会话是否存在
        cursor.execute("SELECT student_id FROM sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return {"error": "会话不存在"}

        student_id = result[0]

        # 查询该学生的最新会话风险等级（从内存中获取，需要前端传递）
        # 这里先简单处理：更新会话状态为已结束
        cursor.execute("""
            UPDATE sessions
            SET end_time = datetime('now')
            WHERE session_id = ?
        """, (session_id,))

        conn.commit()
        conn.close()

        return {"success": True, "message": "会话已保存"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# 会话管理 API（保存会话数据）
# =============================================================================
@app.post("/api/session/end")
async def end_session_post(request: dict):
    """
    结束会话并保存评估数据

    Args:
        request: 包含 session_id

    Returns:
        保存结果
    """
    try:
        session_id = request.get("session_id")

        if not session_id or session_id == "default":
            return {"success": True, "message": "无效session_id"}

        # 从内存中获取会话状态
        hazel = get_shared_hazel_instance()
        session = hazel.get_session(session_id)

        turn_count = session.turn_count
        risk_level = session.risk_level.name if hasattr(session.risk_level, 'name') else str(session.risk_level)

        # 获取评估数据
        assessments = {}
        if hasattr(session, 'assessment') and session.assessment:
            for dim, dim_obj in session.assessment.dimensions.items():
                if dim_obj.is_assessed and dim_obj.score is not None:
                    assessments[dim.name] = dim_obj.score

        import sqlite3
        conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
        cursor = conn.cursor()

        # 只对 MEDIUM/HIGH 风险学生保存评估数据
        if risk_level in ["MEDIUM", "HIGH"] and assessments:
            # 更新会话风险等级
            cursor.execute("""
                UPDATE sessions
                SET risk_level = ?, end_time = datetime('now')
                WHERE session_id = ?
            """, (risk_level, session_id))

            # 保存评估数据
            for dimension, score in assessments.items():
                cursor.execute("""
                    INSERT INTO assessments (session_id, dimension, score, final_score)
                    VALUES (?, ?, ?, ?)
                """, (session_id, dimension, score, score))

            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": f"高风险会话已保存，风险等级: {risk_level}",
                "risk_level": risk_level,
                "assessments": assessments
            }
        else:
            # LOW 风险只标记结束时间，不保存其他数据
            cursor.execute("""
                UPDATE sessions
                SET end_time = datetime('now')
                WHERE session_id = ?
            """, (session_id,))

            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": "会话已结束（低风险，不保存详细数据）",
                "risk_level": risk_level
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/session/end")
async def end_session_get(session_id: str):
    """
    结束会话并保存评估数据（GET版本，用于页面关闭时调用）
    """
    return await end_session_post({"session_id": session_id})


# 教师登录请求模型
class TeacherLoginRequest(BaseModel):
    teacher_id: str
    password: str


@app.post("/api/teacher/verify")
async def verify_teacher(request: TeacherLoginRequest):
    """
    验证教师账号

    Args:
        request: 教师ID和密码

    Returns:
        教师信息
    """
    teacher = TeacherManager.verify_teacher(request.teacher_id)

    if not teacher:
        return {
            "valid": False,
            "message": "教师ID无效或不存在"
        }

    # 验证密码
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM teachers WHERE teacher_id = ?", (request.teacher_id,))
    row = cursor.fetchone()
    conn.close()

    if not row or row[0] != request.password:
        return {
            "valid": False,
            "message": "密码错误，请检查后重试"
        }

    return {
        "valid": True,
        "teacher_id": teacher["teacher_id"],
        "name": teacher["name"],
        "role": teacher["role"],
        "grade": teacher["grade"],
        "class": teacher["class"]
    }


# 学生修改密码请求模型
class ChangePasswordRequest(BaseModel):
    student_id: str
    old_password: str
    new_password: str


@app.post("/api/auth/change-password")
async def change_student_password(request: ChangePasswordRequest):
    """
    学生修改密码
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 验证旧密码
    cursor.execute("SELECT password FROM students WHERE student_id = ?", (request.student_id,))
    row = cursor.fetchone()

    if not row or row[0] != request.old_password:
        conn.close()
        return {"success": False, "message": "原密码错误"}

    # 更新密码
    cursor.execute("UPDATE students SET password = ? WHERE student_id = ?",
                  (request.new_password, request.student_id))
    conn.commit()
    conn.close()

    return {"success": True, "message": "密码修改成功"}


# 教师修改密码请求模型
class TeacherChangePasswordRequest(BaseModel):
    teacher_id: str
    old_password: str
    new_password: str


@app.post("/api/teacher/change-password")
async def change_teacher_password(request: TeacherChangePasswordRequest):
    """
    教师修改密码
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 验证旧密码
    cursor.execute("SELECT password FROM teachers WHERE teacher_id = ?", (request.teacher_id,))
    row = cursor.fetchone()

    if not row or row[0] != request.old_password:
        conn.close()
        return {"success": False, "message": "原密码错误"}

    # 更新密码
    cursor.execute("UPDATE teachers SET password = ? WHERE teacher_id = ?",
                  (request.new_password, request.teacher_id))
    conn.commit()
    conn.close()

    return {"success": True, "message": "密码修改成功"}


# 重置学生密码请求模型
class ResetStudentPasswordRequest(BaseModel):
    teacher_id: str
    student_id: str


@app.post("/api/teacher/reset-student-password")
async def reset_student_password(request: ResetStudentPasswordRequest):
    """
    老师重置学生密码（重置为学号）
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 验证学生是否存在
    cursor.execute("SELECT student_id FROM students WHERE student_id = ?", (request.student_id,))
    if not cursor.fetchone():
        conn.close()
        return {"success": False, "message": "学生不存在"}

    # 重置密码为学号
    cursor.execute("UPDATE students SET password = ? WHERE student_id = ?",
                  (request.student_id, request.student_id))
    conn.commit()
    conn.close()

    return {"success": True, "message": f"密码已重置为学号 {request.student_id}"}


# 重置老师密码请求模型
class ResetTeacherPasswordRequest(BaseModel):
    admin_teacher_id: str  # 心理老师ID
    target_teacher_id: str


@app.post("/api/teacher/reset-teacher-password")
async def reset_teacher_password(request: ResetTeacherPasswordRequest):
    """
    心理老师重置普通老师密码（重置为教师ID）
    只有心理老师(role=psychologist)才能执行
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 验证权限：只有心理老师可以重置
    cursor.execute("SELECT role FROM teachers WHERE teacher_id = ?", (request.admin_teacher_id,))
    admin_row = cursor.fetchone()
    if not admin_row or admin_row[0] != 'psychologist':
        conn.close()
        return {"success": False, "message": "只有心理老师才能重置密码"}

    # 验证目标老师是否存在
    cursor.execute("SELECT teacher_id FROM teachers WHERE teacher_id = ?", (request.target_teacher_id,))
    if not cursor.fetchone():
        conn.close()
        return {"success": False, "message": "教师不存在"}

    # 重置密码为教师ID
    cursor.execute("UPDATE teachers SET password = ? WHERE teacher_id = ?",
                  (request.target_teacher_id, request.target_teacher_id))
    conn.commit()
    conn.close()

    return {"success": True, "message": f"密码已重置为教师ID {request.target_teacher_id}"}


# =============================================================================
# 密码重置申请相关API
# =============================================================================

class RequestPasswordResetRequest(BaseModel):
    user_id: str
    user_type: str = "student"  # student or teacher


@app.post("/api/auth/request-password-reset")
async def request_password_reset(request: RequestPasswordResetRequest):
    """
    用户提交密码重置申请（学生或老师）
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    user_id = request.user_id.strip()

    if request.user_type == "student":
        # 验证学生是否存在
        cursor.execute("SELECT student_id FROM students WHERE student_id = ?", (user_id,))
        if not cursor.fetchone():
            conn.close()
            return {"success": False, "message": "学号不存在"}

        # 检查是否已有待处理的申请
        cursor.execute(
            "SELECT id FROM password_reset_requests WHERE user_id = ? AND user_type = 'student' AND status = 'pending'",
            (user_id,)
        )
        if cursor.fetchone():
            conn.close()
            return {"success": False, "message": "已有待处理的申请，请耐心等待老师处理"}

        # 创建新申请
        cursor.execute(
            "INSERT INTO password_reset_requests (user_id, user_type, status) VALUES (?, 'student', 'pending')",
            (user_id,)
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "申请已提交，请等待老师审核"}

    elif request.user_type == "teacher":
        # 验证老师是否存在
        cursor.execute("SELECT teacher_id FROM teachers WHERE teacher_id = ?", (user_id,))
        if not cursor.fetchone():
            conn.close()
            return {"success": False, "message": "教师ID不存在"}

        # 检查是否已有待处理的申请
        cursor.execute(
            "SELECT id FROM password_reset_requests WHERE user_id = ? AND user_type = 'teacher' AND status = 'pending'",
            (user_id,)
        )
        if cursor.fetchone():
            conn.close()
            return {"success": False, "message": "已有待处理的申请，请耐心等待心理老师处理"}

        # 创建新申请
        cursor.execute(
            "INSERT INTO password_reset_requests (user_id, user_type, status) VALUES (?, 'teacher', 'pending')",
            (user_id,)
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "申请已提交，请等待心理老师审核"}

    conn.close()
    return {"success": False, "message": "无效的用户类型"}


@app.get("/api/teacher/password-reset-requests")
async def get_password_reset_requests(teacher_id: str):
    """
    老师获取待处理的密码重置申请列表
    心理老师可以看到所有老师的重置申请
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 验证老师权限
    cursor.execute("SELECT role, grade, class FROM teachers WHERE teacher_id = ?", (teacher_id,))
    teacher_row = cursor.fetchone()

    if not teacher_row:
        conn.close()
        return {"error": "教师不存在"}

    teacher_role, teacher_grade, teacher_class = teacher_row

    requests = []

    if teacher_role == 'psychologist':
        # 心理老师：获取所有老师的重置申请
        cursor.execute("""
            SELECT pr.id, pr.user_id, pr.user_type, pr.status, pr.created_at, t.name
            FROM password_reset_requests pr
            JOIN teachers t ON pr.user_id = t.teacher_id
            WHERE pr.status = 'pending'
            AND pr.user_type = 'teacher'
            ORDER BY pr.created_at DESC
        """)

        for row in cursor.fetchall():
            requests.append({
                "id": row[0],
                "user_id": row[1],
                "user_type": row[2],
                "status": row[3],
                "created_at": row[4],
                "user_name": row[5] or ""
            })

        # 同时获取学生的重置申请
        cursor.execute("""
            SELECT pr.id, pr.user_id, pr.user_type, pr.status, pr.created_at, s.name
            FROM password_reset_requests pr
            JOIN students s ON pr.user_id = s.student_id
            WHERE pr.status = 'pending'
            AND pr.user_type = 'student'
            ORDER BY pr.created_at DESC
        """)

        for row in cursor.fetchall():
            requests.append({
                "id": row[0],
                "user_id": row[1],
                "user_type": row[2],
                "status": row[3],
                "created_at": row[4],
                "user_name": row[5] or ""
            })
    else:
        # 普通老师：获取该老师班级的学生重置申请
        cursor.execute("""
            SELECT pr.id, pr.user_id, pr.user_type, pr.status, pr.created_at, s.name
            FROM password_reset_requests pr
            JOIN students s ON pr.user_id = s.student_id
            WHERE pr.status = 'pending'
            AND pr.user_type = 'student'
            AND s.grade = ?
            AND s.class = ?
            ORDER BY pr.created_at DESC
        """, (teacher_grade, teacher_class))

        for row in cursor.fetchall():
            requests.append({
                "id": row[0],
                "user_id": row[1],
                "user_type": row[2],
                "status": row[3],
                "created_at": row[4],
                "user_name": row[5] or ""
            })

    conn.close()
    return {"requests": requests}


class ApprovePasswordResetRequest(BaseModel):
    teacher_id: str
    request_id: int


@app.post("/api/teacher/approve-password-reset")
async def approve_password_reset(request: ApprovePasswordResetRequest):
    """
    老师批准密码重置申请，重置学生或老师的密码
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 获取申请信息
    cursor.execute(
        "SELECT user_id, user_type, status FROM password_reset_requests WHERE id = ?",
        (request.request_id,)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"success": False, "message": "申请不存在"}

    user_id, user_type, status = row

    if status != 'pending':
        conn.close()
        return {"success": False, "message": "申请已被处理"}

    if user_type == 'student':
        # 重置学生密码为学号
        cursor.execute(
            "UPDATE students SET password = ? WHERE student_id = ?",
            (user_id, user_id)
        )
        message = f"密码已重置为学号 {user_id}"
    elif user_type == 'teacher':
        # 重置老师密码为教师ID
        cursor.execute(
            "UPDATE teachers SET password = ? WHERE teacher_id = ?",
            (user_id, user_id)
        )
        message = f"密码已重置为教师ID {user_id}"
    else:
        conn.close()
        return {"success": False, "message": "无效的用户类型"}

    # 更新申请状态为已批准
    cursor.execute(
        "UPDATE password_reset_requests SET status = 'approved' WHERE id = ?",
        (request.request_id,)
    )

    conn.commit()
    conn.close()

    return {"success": True, "message": message}


@app.get("/api/teacher/teachers")
async def get_all_teachers(teacher_id: str):
    """
    获取所有老师列表（仅心理老师可用）
    """
    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 验证是否是心理老师
    cursor.execute("SELECT role FROM teachers WHERE teacher_id = ?", (teacher_id,))
    admin_row = cursor.fetchone()
    if not admin_row or admin_row[0] != 'psychologist':
        conn.close()
        return {"error": "只有心理老师才能查看"}

    # 获取所有老师
    cursor.execute("""
        SELECT teacher_id, name, role, grade, class, is_active
        FROM teachers
        ORDER BY teacher_id
    """)
    teachers = []
    for row in cursor.fetchall():
        teachers.append({
            "teacher_id": row[0],
            "name": row[1],
            "role": row[2],
            "grade": row[3],
            "class": row[4],
            "is_active": row[5]
        })

    conn.close()
    return {"teachers": teachers}


@app.get("/api/teacher/classes")
async def get_teacher_classes(teacher_id: str):
    """
    获取教师可查看的班级列表

    Args:
        teacher_id: 教师ID

    Returns:
        班级列表
    """
    teacher = TeacherManager.verify_teacher(teacher_id)

    if not teacher:
        return {"error": "教师ID无效"}

    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 心理老师可以查看所有班级
    if teacher["role"] == "psychologist":
        cursor.execute("""
            SELECT grade, class,
                   COUNT(DISTINCT st.student_id) as student_count,
                   COUNT(DISTINCT s.session_id) as session_count,
                   SUM(CASE WHEN s.risk_level = 'LOW' THEN 1 ELSE 0 END) as low_count,
                   SUM(CASE WHEN s.risk_level = 'MEDIUM' THEN 1 ELSE 0 END) as medium_count,
                   SUM(CASE WHEN s.risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_count
            FROM students st
            LEFT JOIN sessions s ON st.student_id = s.student_id
            WHERE st.is_active = 1
            GROUP BY grade, class
            ORDER BY grade, class
        """)
    else:
        # 普通老师只能看自己班级
        cursor.execute("""
            SELECT grade, class,
                   COUNT(DISTINCT st.student_id) as student_count,
                   COUNT(DISTINCT s.session_id) as session_count,
                   SUM(CASE WHEN s.risk_level = 'LOW' THEN 1 ELSE 0 END) as low_count,
                   SUM(CASE WHEN s.risk_level = 'MEDIUM' THEN 1 ELSE 0 END) as medium_count,
                   SUM(CASE WHEN s.risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_count
            FROM students st
            LEFT JOIN sessions s ON st.student_id = s.student_id
            WHERE st.is_active = 1 AND st.grade = ? AND st.class = ?
            GROUP BY grade, class
            ORDER BY grade, class
        """, (teacher["grade"], teacher["class"]))

    results = cursor.fetchall()
    conn.close()

    classes = []
    for r in results:
        classes.append({
            "grade": r[0],
            "class": r[1],
            "student_count": r[2] or 0,
            "session_count": r[3] or 0,
            "risk_low": r[4] or 0,
            "risk_medium": r[5] or 0,
            "risk_high": r[6] or 0
        })

    return {
        "role": teacher["role"],
        "teacher_name": teacher["name"],
        "classes": classes
    }


@app.get("/api/teacher/students")
async def get_class_students(teacher_id: str, grade: str, class_num: str):
    """
    获取班级学生列表

    Args:
        teacher_id: 教师ID
        grade: 年级
        class_num: 班级

    Returns:
        学生列表
    """
    teacher = TeacherManager.verify_teacher(teacher_id)

    if not teacher:
        return {"error": "教师ID无效"}

    # 检查权限：普通老师只能看自己班级
    if teacher["role"] == "normal" and (teacher["grade"] != grade or teacher["class"] != class_num):
        return {"error": "无权查看其他班级"}

    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 获取学生列表及最新会话风险等级
    cursor.execute("""
        SELECT st.student_id, st.name,
               s.session_id, s.risk_level, s.end_time,
               a.final_score,
               (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) as msg_count
        FROM students st
        LEFT JOIN (
            SELECT student_id, session_id, risk_level, end_time, start_time,
                   ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY start_time DESC) as rn
            FROM sessions
        ) s ON st.student_id = s.student_id AND s.rn = 1
        LEFT JOIN assessments a ON s.session_id = a.session_id AND a.dimension = 'overall'
        WHERE st.is_active = 1 AND st.grade = ? AND st.class = ?
        ORDER BY st.student_id
    """, (grade, class_num))

    results = cursor.fetchall()
    conn.close()

    students = []
    for r in results:
        risk_level = r[3] if r[3] else 'LOW'
        students.append({
            "student_id": r[0],
            "name": r[1] or '-',
            "session_id": r[2],
            "risk_level": risk_level,
            "turn_count": r[6] if r[6] else 0,
            "has_session": r[2] is not None,
            "final_score": r[5]
        })

    return {
        "grade": grade,
        "class": class_num,
        "role": teacher["role"],
        "students": students
    }


@app.get("/api/teacher/class-stats")
async def get_class_stats(teacher_id: str, grade: str, class_num: str):
    """
    获取班级的统计数据（饼图和趋势数据）

    Returns:
        风险分布和每周会话趋势
    """
    import sqlite3
    from datetime import datetime, timedelta

    # 验证教师权限
    teacher = TeacherManager.verify_teacher(teacher_id)
    if not teacher:
        return {"error": "教师ID无效"}

    # 普通老师只能看自己班级
    if teacher["role"] == "normal" and (teacher["grade"] != grade or teacher["class"] != class_num):
        return {"error": "无权查看其他班级"}

    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 1. 风险等级分布（按学生统计，取每个学生的最新会话风险等级，没有会话的也算正常）
    cursor.execute("""
        SELECT
            SUM(CASE WHEN latest_risk = 'LOW' OR latest_risk IS NULL THEN 1 ELSE 0 END) as low,
            SUM(CASE WHEN latest_risk = 'MEDIUM' THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN latest_risk = 'HIGH' THEN 1 ELSE 0 END) as high,
            COUNT(*) as total
        FROM (
            SELECT
                st.student_id,
                s.risk_level as latest_risk,
                ROW_NUMBER() OVER (PARTITION BY st.student_id ORDER BY s.start_time DESC) as rn
            FROM students st
            LEFT JOIN sessions s ON st.student_id = s.student_id
            WHERE st.grade = ? AND st.class = ?
        ) t
        WHERE rn = 1
    """, (grade, class_num))

    risk_data = cursor.fetchone()
    risk_distribution = [
        {"name": "正常", "value": risk_data[0] or 0, "color": "#22c55e"},
        {"name": "需关注", "value": risk_data[1] or 0, "color": "#eab308"},
        {"name": "高风险", "value": risk_data[2] or 0, "color": "#ef4444"}
    ]

    # 2. 按会话次数的趋势（横轴：次数，纵轴：评分）
    cursor.execute("""
        SELECT
            s.session_id,
            s.start_time,
            (SELECT COUNT(*) FROM sessions s2
             JOIN students st2 ON s2.student_id = st2.student_id
             WHERE st2.grade = ? AND st2.class = ?
             AND s2.start_time <= s.start_time) as session_order
        FROM sessions s
        JOIN students st ON s.student_id = st.student_id
        WHERE st.grade = ? AND st.class = ?
        ORDER BY s.start_time DESC
        LIMIT 20
    """, (grade, class_num, grade, class_num))

    sessions_data = []
    for row in cursor.fetchall():
        session_id = row[0]

        # 获取该会话的评分（如果有）
        cursor2 = conn.cursor()
        cursor2.execute("""
            SELECT AVG(final_score) FROM assessments
            WHERE session_id = ? AND final_score IS NOT NULL
        """, (session_id,))
        avg_score_row = cursor2.fetchone()
        avg_score = round(avg_score_row[0], 1) if avg_score_row[0] else 0

        sessions_data.append({
            "order": row[2],
            "time": row[1][:16] if row[1] else "",
            "score": avg_score
        })

    sessions_data.reverse()  # 按时间正序

    # 3. 各维度平均分（如果有数据）
    cursor.execute("""
        SELECT dimension, AVG(final_score) as avg_score
        FROM assessments a
        JOIN sessions s ON a.session_id = s.session_id
        JOIN students st ON s.student_id = st.student_id
        WHERE st.grade = ? AND st.class = ?
        AND a.final_score IS NOT NULL
        GROUP BY a.dimension
    """, (grade, class_num))

    dimension_scores = []
    for row in cursor.fetchall():
        dimension_scores.append({"dimension": row[0], "score": round(row[1], 1) if row[1] else 0})

    conn.close()

    return {
        "grade": grade,
        "class": class_num,
        "risk_distribution": risk_distribution,
        "session_trend": sessions_data,
        "dimension_scores": dimension_scores
    }


@app.get("/api/teacher/assessment-advice")
async def get_assessment_advice(session_id: str, teacher_id: str = None, teacher_role: str = None):
    """
    获取学生评估AI建议（带权限控制）

    Args:
        session_id: 会话ID
        teacher_id: 教师ID（可选，用于权限验证）
        teacher_role: 教师角色（"normal"或"psychologist"）

    Returns:
        AI生成的建议文本
    """
    # 如果提供了教师ID，进行权限验证
    if teacher_id and teacher_role == "normal":
        # 查询该会话的风险等级
        import sqlite3
        conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
        cursor = conn.cursor()
        cursor.execute("SELECT risk_level FROM sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0] not in ["MEDIUM", "HIGH"]:
            raise HTTPException(status_code=403, detail="普通老师只能查看需要关注的学生")
    try:
        # 查询会话信息
        import sqlite3
        conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT s.student_id, st.name, s.turn_count, s.stage, s.risk_level
            FROM sessions s
            LEFT JOIN students st ON s.student_id = st.student_id
            WHERE s.session_id = ?
        """, (session_id,))

        session_info = cursor.fetchone()
        if not session_info:
            conn.close()
            raise HTTPException(status_code=404, detail="Session not found")

        student_id, student_name, turn_count, stage, risk_level = session_info

        # 查询各维度评估分数
        cursor.execute("""
            SELECT dimension, final_score
            FROM assessments
            WHERE session_id = ? AND final_score IS NOT NULL
        """, (session_id,))

        assessments = cursor.fetchall()
        conn.close()

        # 构建评估数据描述
        assessment_text = ""
        dimension_scores = {}
        if assessments and len(assessments) > 0:
            assessment_text = "各维度评估分数：\n"
            for dim, score in assessments:
                assessment_text += f"- {dim}: {score:.2f}\n"
                dimension_scores[dim] = score
        else:
            assessment_text = "暂无详细评估数据"

        # 调用LLM生成建议
        hazel = get_shared_hazel_instance()

        # 加载 suggestion_prompt.yaml
        import yaml
        from pathlib import Path
        SUGGESTION_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "suggestion_prompt.yaml"

        if SUGGESTION_PROMPT_PATH.exists():
            with open(SUGGESTION_PROMPT_PATH, 'r', encoding='utf-8') as f:
                prompt_config = yaml.safe_load(f)

            # 根据是否有评估数据选择模板
            if assessments and len(assessments) > 0:
                prompt_template = prompt_config.get("user_prompt_template", "")
            else:
                prompt_template = prompt_config.get("no_assessment_prompt", "")

            # 填充模板
            prompt = prompt_template.format(
                student_id=student_id or '',
                student_name=student_name or '',
                turn_count=turn_count or 0,
                stage=stage or '未知',
                risk_level=risk_level or '未知',
                assessment_text=assessment_text
            )
        else:
            # 如果yaml文件不存在，使用简单的默认值
            prompt = f"学生学号{student_id}，风险等级{risk_level}，请给出教师建议。"

        # 调用LLM
        response = hazel.llm_engine.generate_brain_response(prompt)

        # 尝试解析JSON
        try:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            advice_data = json.loads(response)
            advice_data["dimension_scores"] = dimension_scores
            return {
                "student_id": student_id,
                "student_name": student_name,
                "risk_level": risk_level,
                "turn_count": turn_count or 0,
                "advice": advice_data
            }
        except json.JSONDecodeError:
            # 如果解析失败，返回原始文本
            return {
                "student_id": student_id,
                "student_name": student_name,
                "risk_level": risk_level,
                "turn_count": turn_count or 0,
                "advice": {
                    "overall_assessment": response,
                    "risk_concerns": [],
                    "teacher_advice": "",
                    "need_referral": "unknown",
                    "referral_reason": "",
                    "dimension_scores": dimension_scores
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teacher/session-messages")
async def get_session_messages(session_id: str, teacher_role: str):
    """
    获取会话对话记录（带权限控制）

    Args:
        session_id: 会话ID
        teacher_role: 教师角色（"normal"或"psychologist"）

    Returns:
        对话记录（心理老师只能看MEDIUM+HIGH风险的会话）
    """
    try:
        import sqlite3
        conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
        cursor = conn.cursor()

        # 查询会话风险等级
        cursor.execute("""
            SELECT risk_level, s.student_id, st.name
            FROM sessions s
            LEFT JOIN students st ON s.student_id = st.student_id
            WHERE s.session_id = ?
        """, (session_id,))

        session_info = cursor.fetchone()
        if not session_info:
            conn.close()
            raise HTTPException(status_code=404, detail="Session not found")

        risk_level, student_id, student_name = session_info

        # 权限检查
        if teacher_role == "normal":
            # 普通老师不能看对话
            conn.close()
            raise HTTPException(status_code=403, detail="普通老师无权查看对话内容")

        elif teacher_role == "psychologist":
            # 心理老师只能看MEDIUM+HIGH
            if risk_level not in ["MEDIUM", "HIGH"]:
                conn.close()
                raise HTTPException(status_code=403, detail=f"该学生风险等级为{risk_level or '未知'}，仅MEDIUM+HIGH可查看")

        # 查询对话记录
        cursor.execute("""
            SELECT role, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))

        messages = cursor.fetchall()
        conn.close()

        return {
            "session_id": session_id,
            "student_id": student_id,
            "student_name": student_name,
            "risk_level": risk_level,
            "messages": [
                {
                    "role": msg[0],
                    "content": msg[1],
                    "timestamp": msg[2]
                }
                for msg in messages
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teacher/high-risk-students")
async def get_high_risk_students(teacher_id: str):
    """
    获取高危学生列表（MEDIUM+HIGH风险）

    Returns:
        高危学生列表，按风险等级和时间排序
    """
    teacher = TeacherManager.verify_teacher(teacher_id)

    if not teacher:
        return {"error": "教师ID无效"}

    if teacher["role"] != "psychologist":
        return {"error": "权限不足"}

    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT st.student_id, st.name, st.grade, st.class,
               s.session_id, s.risk_level, s.end_time,
               (SELECT AVG(final_score) FROM assessments a WHERE a.session_id = s.session_id) as avg_score
        FROM students st
        INNER JOIN (
            SELECT student_id, MAX(end_time) as max_time
            FROM sessions
            WHERE risk_level IN ('MEDIUM', 'HIGH')
            GROUP BY student_id
        ) latest ON st.student_id = latest.student_id
        INNER JOIN sessions s ON st.student_id = s.student_id AND s.end_time = latest.max_time
        WHERE st.is_active = 1
        ORDER BY
            CASE s.risk_level
                WHEN 'HIGH' THEN 1
                WHEN 'MEDIUM' THEN 2
                ELSE 3
            END,
            s.end_time DESC
    """)

    results = cursor.fetchall()
    conn.close()

    students = []
    for r in results:
        students.append({
            "student_id": r[0],
            "name": r[1],
            "grade": r[2],
            "class": r[3],
            "latest_session_id": r[4],
            "risk_level": r[5],
            "latest_time": r[6],
            "avg_score": round(r[7], 2) if r[7] else None
        })

    return {"students": students}

@app.get("/api/teacher/assessment-trend")
async def get_assessment_trend(teacher_id: str):
    """
    获取评估趋势数据（按第n次评估的平均分）

    Returns:
        按评估次数分组的平均综合分数
    """
    teacher = TeacherManager.verify_teacher(teacher_id)

    if not teacher:
        return {"error": "教师ID无效"}

    if teacher["role"] != "psychologist":
        return {"error": "权限不足"}

    import sqlite3
    conn = sqlite3.connect("/home/dazzle/Hazel_AI/hazel_ai.db")
    cursor = conn.cursor()

    # 计算每个session的综合分数（加权平均）
    cursor.execute("""
        WITH session_scores AS (
            SELECT
                a.session_id,
                s.student_id,
                SUM(CASE
                    WHEN a.dimension = 'PHYSICAL_FUNCTION' THEN a.final_score * 0.30
                    WHEN a.dimension = 'COGNITIVE_DISTORTION' THEN a.final_score * 0.25
                    WHEN a.dimension = 'ACADEMIC_PRESSURE' THEN a.final_score * 0.20
                    WHEN a.dimension = 'EMOTIONAL_RESILIENCE' THEN a.final_score * 0.15
                    WHEN a.dimension = 'SOCIAL_SUPPORT' THEN a.final_score * 0.10
                    ELSE 0
                END) as weighted_score
            FROM assessments a
            JOIN sessions s ON a.session_id = s.session_id
            WHERE a.final_score IS NOT NULL
            GROUP BY a.session_id, s.student_id
        ),
        assessment_order AS (
            SELECT
                student_id,
                session_id,
                weighted_score,
                ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY session_id) as assessment_num
            FROM session_scores
        )
        SELECT
            assessment_num,
            AVG(weighted_score) as avg_score,
            COUNT(*) as count
        FROM assessment_order
        GROUP BY assessment_num
        ORDER BY assessment_num
    """)

    results = cursor.fetchall()
    conn.close()

    trend_data = []
    for r in results:
        trend_data.append({
            "assessment_num": r[0],
            "avg_score": round(r[1], 2) if r[1] else 0,
            "count": r[2]
        })

    return {"trend": trend_data}

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天API（支持多气泡输出）

    Returns:
        Server-Sent Events (SSE) 流式响应
        格式：data: {"content": "..."} 或 data: {"split": true}
    """
    import time
    try:
        req_start = time.time()
        # 获取最后一条用户消息
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        print(f"🎯 [T+0ms] chat_stream 被调用")
        print(f"📨 收到消息: {user_message[:50]}...")

        # 获取共享Hazel实例（可能抛出"系统正在初始化"异常）
        try:
            hazel = get_shared_hazel_instance()
        except Exception as e:
            if "系统正在初始化" in str(e):
                # 模型正在加载中
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=503,
                    content={"error": "系统正在初始化，请稍候... (首次加载模型需要1-2分钟)"}
                )
            else:
                raise
        print(f"⏱️ [T+{int((time.time()-req_start)*1000)}ms] Hazel实例获取完成")

        # 获取该用户的独立SessionState
        session_id = request.session_id or "default"

        # 获取session的student_id
        session_info = sessions.get(session_id, {})
        student_id = session_info.get("student_id", session_id)

        user_session = hazel.get_session(session_id, student_id)
        print(f"⏱️ [T+{int((time.time()-req_start)*1000)}ms] SessionState获取完成 (session_id={session_id}, user_id={student_id})")

        def generate():
            """生成SSE流（使用同步生成器）"""
            try:
                print(f"🔄 [T+{int((time.time()-req_start)*1000)}ms] generate() 被调用")
                chunk_count = 0

                avatar_start = time.time()
                print(f"🔍 [T+{int((time.time()-req_start)*1000)}ms] 准备调用 avatar.chat()...")
                # 调用avatar.chat()生成器，使用用户独立的session
                chat_generator = hazel.avatar.chat(user_session, user_message)
                print(f"⏱️ [T+{int((time.time()-req_start)*1000)}ms] avatar.chat() 返回生成器，耗时: {int((time.time()-avatar_start)*1000)}ms")

                # 初始化气泡分割器
                splitter = StreamBubbleSplitter(min_bubble_length=5, max_bubbles=3)

                first_chunk = True
                for text_chunk in chat_generator:
                    if first_chunk:
                        print(f"⏱️ [T+{int((time.time()-req_start)*1000)}ms] 第一个chunk yield，总耗时: {int((time.time()-req_start)*1000)}ms")
                        first_chunk = False

                    chunk_count += 1
                    print(f"📦 [T+{int((time.time()-req_start)*1000)}ms] Chunk #{chunk_count}: {repr(text_chunk[:50])}")

                    # 使用分割器处理chunk
                    parts = splitter.process(text_chunk)

                    for part in parts:
                        if part == "[SPLIT]":
                            # 发送分割信号
                            yield f"data: {json.dumps({'split': True}, ensure_ascii=False)}\n\n"
                        else:
                            # 发送内容
                            yield f"data: {json.dumps({'content': part}, ensure_ascii=False)}\n\n"

                print(f"✅ [T+{int((time.time()-req_start)*1000)}ms] 流式生成完成，共 {chunk_count} 个chunk")

                # 流结束，发送剩余内容
                remaining = splitter.finalize()
                for part in remaining:
                    yield f"data: {json.dumps({'content': part}, ensure_ascii=False)}\n\n"

                # 发送结束信号
                yield f"data: [DONE]\n\n"

            except GeneratorExit:
                print("⚠️  客户端断开连接")
                raise
            except Exception as e:
                import traceback
                print(f"❌ 生成流时出错: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SFBT 调试 API
# =============================================================================

import yaml
from pathlib import Path

SFBT_PROMPT_PATH = Path(__file__).parent.parent / "sfbt" / "sfbt_prompt.yaml"

class SFBTPromptRequest(BaseModel):
    method: str
    persona: str
    user_message: str
    history: Optional[List[Message]] = []

@app.get("/api/sfbt/prompts")
async def get_sfbt_prompts():
    """获取所有 SFBT 干预方法的 prompt 模板"""
    try:
        if not SFBT_PROMPT_PATH.exists():
            return {"error": "sfbt_prompt.yaml 不存在"}

        with open(SFBT_PROMPT_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return {
            "persona": data.get("persona", ""),
            "methods": data.get("methods", {}),
            "common_rules": data.get("common_rules", "")
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/sfbt/prompts")
async def save_sfbt_prompts(data: dict):
    """保存 SFBT 干预方法的 prompt 模板"""
    try:
        with open(SFBT_PROMPT_PATH, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/sfbt/test")
async def test_sfbt_prompt(request: SFBTPromptRequest):
    """测试 SFBT 干预方法 - 组合 prompt 并调用 Avatar"""
    try:
        # 加载 scaffold
        with open(SFBT_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)

        method_scaffold = prompt_data.get("methods", {}).get(request.method, "")
        common_rules = prompt_data.get("common_rules", "")

        # 组合完整 prompt
        system_prompt = f"{request.persona}\n\n{method_scaffold}\n\n{common_rules}"

        # 格式化历史
        history_text = ""
        for msg in request.history:
            history_text += f"{msg.role}: {msg.content}\n"

        user_prompt = f"""对话历史：
{history_text}

当前用户消息：{request.user_message}"""

        # 获取共享 Hazel 实例
        hazel = get_shared_hazel_instance()

        # 直接调用 avatar 的 LLM（不经过完整流程）
        from core.llm_engine import LLEngine
        if isinstance(hazel.avatar.llm_engine, LLEngine):
            response = hazel.avatar.llm_engine.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
        else:
            # 兼容其他引擎
            response = "当前引擎不支持直接调用"

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# =============================================================================
# 启动命令
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 启动 Hazel AI FastAPI 后端...")
    print("=" * 60)
    print("📝 API地址：http://localhost:8000")
    print("📚 API文档：http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
