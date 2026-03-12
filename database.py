"""
SQLite 数据库管理模块
用于学生验证和会话记录
"""
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import json

# =============================================================================
# 数据库配置
# =============================================================================
DB_PATH = Path(__file__).parent / "hazel_ai.db"

# =============================================================================
# 数据库初始化
# =============================================================================
def init_database():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 学生表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT,
            grade TEXT,
            class TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)

    # 会话表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            student_id TEXT NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            turn_count INTEGER DEFAULT 0,
            stage TEXT DEFAULT 'WARM_UP_SCAN',
            risk_level TEXT DEFAULT 'LOW',
            exit_reason TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    # 对话记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dimension TEXT,
            risk_score REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # 评估结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            dimension TEXT NOT NULL,
            score REAL NOT NULL,
            assessment_count INTEGER DEFAULT 0,
            final_score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # 教师表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT UNIQUE NOT NULL,
            name TEXT,
            role TEXT NOT NULL CHECK(role IN ('normal', 'psychologist')),
            grade TEXT,
            class TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ 数据库初始化完成：{DB_PATH}")


# =============================================================================
# 学生管理
# =============================================================================
class StudentManager:
    """学生信息管理"""

    @staticmethod
    def add_student(student_id: str, name: str = None, grade: str = None, class_: str = None) -> bool:
        """
        添加学生（由教师预先导入）

        Args:
            student_id: 学号
            name: 姓名
            grade: 年级
            class_: 班级

        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO students (student_id, name, grade, class)
                VALUES (?, ?, ?, ?)
            """, (student_id, name, grade, class_))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 添加学生失败：{e}")
            return False

    @staticmethod
    def verify_student(student_id: str) -> Optional[Dict]:
        """
        验证学生是否存在

        Args:
            student_id: 学号

        Returns:
            学生信息字典，如果不存在返回 None
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT student_id, name, grade, class, is_active
                FROM students
                WHERE student_id = ? AND is_active = 1
            """, (student_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    "student_id": result[0],
                    "name": result[1],
                    "grade": result[2],
                    "class": result[3]
                }
            return None
        except Exception as e:
            print(f"❌ 验证学生失败：{e}")
            return None

    @staticmethod
    def get_all_students() -> List[Dict]:
        """获取所有学生列表"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT student_id, name, grade, class, is_active
                FROM students
                WHERE is_active = 1
                ORDER BY grade, class, student_id
            """)

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    "student_id": r[0],
                    "name": r[1],
                    "grade": r[2],
                    "class": r[3]
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ 获取学生列表失败：{e}")
            return []

    @staticmethod
    def batch_import(students: List[Dict]) -> int:
        """
        批量导入学生

        Args:
            students: 学生列表，每个学生包含 student_id, name, grade, class

        Returns:
            成功导入的数量
        """
        count = 0
        for student in students:
            if StudentManager.add_student(
                student_id=student.get("student_id"),
                name=student.get("name"),
                grade=student.get("grade"),
                class_=student.get("class")
            ):
                count += 1
        return count


# =============================================================================
# 会话管理
# =============================================================================
class SessionManagerDB:
    """会话记录管理"""

    @staticmethod
    def create_session(session_id: str, student_id: str) -> bool:
        """创建新会话（如果不存在）"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 先检查session是否已存在
            cursor.execute("""
                SELECT session_id FROM sessions WHERE session_id = ?
            """, (session_id,))

            if cursor.fetchone():
                # session已存在，不需要创建
                conn.close()
                return True

            # session不存在，创建新记录
            cursor.execute("""
                INSERT INTO sessions (session_id, student_id)
                VALUES (?, ?)
            """, (session_id, student_id))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 创建会话失败：{e}")
            return False

    @staticmethod
    def update_session(session_id: str, **kwargs) -> bool:
        """更新会话信息"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 构建更新语句
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [session_id]

            cursor.execute(f"""
                UPDATE sessions
                SET {set_clause}
                WHERE session_id = ?
            """, values)

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 更新会话失败：{e}")
            return False

    @staticmethod
    def end_session(session_id: str, exit_reason: str = None) -> bool:
        """结束会话"""
        return SessionManagerDB.update_session(
            session_id,
            end_time=datetime.now().isoformat(),
            exit_reason=exit_reason
        )

    @staticmethod
    def get_session_info(session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                columns = [
                    "id", "session_id", "student_id", "start_time",
                    "end_time", "turn_count", "stage", "risk_level", "exit_reason"
                ]
                return dict(zip(columns, result))
            return None
        except Exception as e:
            print(f"❌ 获取会话信息失败：{e}")
            return None


# =============================================================================
# 消息记录
# =============================================================================
class MessageLogger:
    """对话记录管理"""

    @staticmethod
    def log_message(session_id: str, role: str, content: str,
                   dimension: str = None, risk_score: float = None) -> bool:
        """记录消息"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO messages
                (session_id, role, content, dimension, risk_score)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, dimension, risk_score))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 记录消息失败：{e}")
            return False

    @staticmethod
    def get_session_messages(session_id: str) -> List[Dict]:
        """获取会话的所有消息"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT role, content, timestamp, dimension, risk_score
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    "role": r[0],
                    "content": r[1],
                    "timestamp": r[2],
                    "dimension": r[3],
                    "risk_score": r[4]
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ 获取消息记录失败：{e}")
            return []


# =============================================================================
# 评估记录
# =============================================================================
class AssessmentLogger:
    """评估结果管理"""

    @staticmethod
    def log_assessment(session_id: str, dimension: str,
                      score: float, assessment_count: int = 0) -> bool:
        """记录评估结果"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO assessments
                (session_id, dimension, score, assessment_count)
                VALUES (?, ?, ?, ?)
            """, (session_id, dimension, score, assessment_count))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 记录评估失败：{e}")
            return False

    @staticmethod
    def update_final_score(session_id: str, dimension: str, final_score: float) -> bool:
        """更新维度最终得分"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE assessments
                SET final_score = ?
                WHERE session_id = ? AND dimension = ?
            """, (final_score, session_id, dimension))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 更新最终得分失败：{e}")
            return False


# =============================================================================
# 教师管理
# =============================================================================
class TeacherManager:
    """教师信息管理"""

    @staticmethod
    def add_teacher(teacher_id: str, name: str = None, role: str = "normal",
                     grade: str = None, class_: str = None) -> bool:
        """
        添加教师

        Args:
            teacher_id: 教师ID (T开头普通老师，P开头心理老师)
            name: 姓名
            role: "normal" 或 "psychologist"
            grade: 所属年级 (普通老师用)
            class_: 所属班级 (普通老师用)
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO teachers (teacher_id, name, role, grade, class)
                VALUES (?, ?, ?, ?, ?)
            """, (teacher_id, name, role, grade, class_))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 添加教师失败：{e}")
            return False

    @staticmethod
    def verify_teacher(teacher_id: str) -> Optional[Dict]:
        """
        验证教师是否存在

        Args:
            teacher_id: 教师ID

        Returns:
            教师信息字典，如果不存在返回 None
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT teacher_id, name, role, grade, class, is_active
                FROM teachers
                WHERE teacher_id = ? AND is_active = 1
            """, (teacher_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    "teacher_id": result[0],
                    "name": result[1],
                    "role": result[2],
                    "grade": result[3],
                    "class": result[4]
                }
            return None
        except Exception as e:
            print(f"❌ 验证教师失败：{e}")
            return None

    @staticmethod
    def get_all_teachers() -> List[Dict]:
        """获取所有教师列表"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT teacher_id, name, role, grade, class, is_active
                FROM teachers
                WHERE is_active = 1
                ORDER BY role, teacher_id
            """)

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    "teacher_id": r[0],
                    "name": r[1],
                    "role": r[2],
                    "grade": r[3],
                    "class": r[4]
                }
                for r in results
            ]
        except Exception as e:
            print(f"❌ 获取教师列表失败：{e}")
            return []


# =============================================================================
# 初始化
# =============================================================================
if __name__ == "__main__":
    # 初始化数据库
    init_database()

    # 添加测试学生（示例）
    test_students = [
        {"student_id": "202301001", "name": "张三", "grade": "2023", "class": "1"},
        {"student_id": "202301002", "name": "李四", "grade": "2023", "class": "1"},
        {"student_id": "202302001", "name": "王五", "grade": "2023", "class": "2"},
    ]

    count = StudentManager.batch_import(test_students)
    print(f"✅ 添加了 {count} 名测试学生")

    # 验证学生
    student = StudentManager.verify_student("202301001")
    print(f"验证结果：{student}")

    # 添加测试教师
    TeacherManager.add_teacher("T001", "张老师", "normal", "2023", "1")
    TeacherManager.add_teacher("P001", "心理李老师", "psychologist", None, None)

    print("\n💡 提示：使用 python teacher_console.py 打开教师后台")
