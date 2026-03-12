"""
教师后台管理工具
用于学生管理、会话查看和数据统计
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json
import requests

# =============================================================================
# 数据库路径
# =============================================================================
DB_PATH = Path(__file__).parent / "hazel_ai.db"
API_BASE = "http://localhost:8000"

# =============================================================================
# 教师控制台
# =============================================================================
class TeacherConsole:
    """教师后台管理控制台"""

    def __init__(self):
        """初始化控制台"""
        if not DB_PATH.exists():
            print("❌ 数据库文件不存在，请先运行主程序初始化数据库。")
            return

        self.running = True
        self.teacher_id = None
        self.teacher_role = None  # "normal" 或 "psychologist"

    def login(self):
        """教师登录"""
        print("\n" + "=" * 60)
        print("🔑 Hazel AI - 教师登录")
        print("=" * 60)

        while True:
            teacher_id = input("\n请输入教师ID: ").strip()
            if not teacher_id:
                print("❌ 教师ID不能为空")
                continue

            # 识别角色
            if teacher_id.startswith("T"):
                self.teacher_role = "normal"
                self.teacher_id = teacher_id
                print(f"✅ 登录成功！角色：普通老师")
                return True
            elif teacher_id.startswith("P"):
                self.teacher_role = "psychologist"
                self.teacher_id = teacher_id
                print(f"✅ 登录成功！角色：心理老师")
                return True
            else:
                print("❌ 无效的教师ID格式（应以T或P开头）")
                retry = input("是否重试？(y/n): ").strip().lower()
                if retry != 'y':
                    return False

    def show_menu(self):
        """显示主菜单（根据角色）"""
        print("\n" + "=" * 60)
        print(f"📊 Hazel AI - 教师后台管理 ({'普通老师' if self.teacher_role == 'normal' else '心理老师'})")
        print("=" * 60)
        print("1. 👥 学生管理")
        print("2. 📊 班级统计")
        print("3. 📈 学生评估（查看AI建议）")

        if self.teacher_role == "psychologist":
            print("4. 💬 高风险对话查看")
            print("5. 📤 导出数据")
        else:
            print("4. 📤 导出数据")

        print("0. 🚪 退出")
        print("=" * 60)

    def run(self):
        """运行控制台"""
        # 先登录
        if not self.login():
            print("\n登录失败，退出程序")
            return

        # 主循环
        while self.running:
            self.show_menu()
            choice = input("\n请选择操作（输入数字）: ").strip()

            if choice == "1":
                self.student_management()
            elif choice == "2":
                self.class_statistics()
            elif choice == "3":
                self.student_assessment()
            elif choice == "4":
                if self.teacher_role == "psychologist":
                    self.high_risk_conversations()
                else:
                    self.export_data()
            elif choice == "5":
                if self.teacher_role == "psychologist":
                    self.export_data()
                else:
                    print("\n❌ 无效选择")
            elif choice == "0":
                print("\n👋 再见！")
                self.running = False
            else:
                print("\n❌ 无效选择，请重新输入。")

    # -------------------------------------------------------------------------
    # 学生管理
    # -------------------------------------------------------------------------
    def student_management(self):
        """学生管理子菜单"""
        while True:
            print("\n" + "-" * 40)
            print("👥 学生管理")
            print("-" * 40)
            print("1. 查看所有学生")
            print("2. 添加单个学生")
            print("3. 批量导入学生")
            print("4. 搜索学生")
            print("0. 返回主菜单")
            print("-" * 40)

            choice = input("\n请选择操作: ").strip()

            if choice == "1":
                self.list_all_students()
            elif choice == "2":
                self.add_single_student()
            elif choice == "3":
                self.batch_import_students()
            elif choice == "4":
                self.search_student()
            elif choice == "0":
                break
            else:
                print("\n❌ 无效选择")

    def list_all_students(self):
        """列出所有学生"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT student_id, name, grade, class,
                       COUNT(DISTINCT s.session_id) as session_count
                FROM students st
                LEFT JOIN sessions s ON st.student_id = s.student_id
                WHERE st.is_active = 1
                GROUP BY st.student_id
                ORDER BY grade, class, student_id
            """)

            results = cursor.fetchall()
            conn.close()

            if not results:
                print("\n📭 暂无学生记录")
                return

            print(f"\n共找到 {len(results)} 名学生：\n")
            print(f"{'学号':<12} {'姓名':<10} {'年级':<8} {'班级':<8} {'会话次数'}")
            print("-" * 60)

            for r in results:
                print(f"{r[0]:<12} {r[1] or '-':<10} {r[2] or '-':<8} {r[3] or '-':<8} {r[4]}")

        except Exception as e:
            print(f"\n❌ 查询失败：{e}")

    def add_single_student(self):
        """添加单个学生"""
        print("\n➕ 添加学生")
        student_id = input("学号: ").strip()
        name = input("姓名（可选）: ").strip() or None
        grade = input("年级（可选）: ").strip() or None
        class_ = input("班级（可选）: ").strip() or None

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO students (student_id, name, grade, class)
                VALUES (?, ?, ?, ?)
            """, (student_id, name, grade, class_))

            conn.commit()
            conn.close()

            print("\n✅ 学生添加成功！")

        except sqlite3.IntegrityError:
            print("\n❌ 该学号已存在！")
        except Exception as e:
            print(f"\n❌ 添加失败：{e}")

    def batch_import_students(self):
        """批量导入学生"""
        print("\n📥 批量导入学生")
        print("格式：学号,姓名,年级,班级（每行一个学生）")
        print("输入完成后输入空行结束\n")

        students = []
        while True:
            line = input().strip()
            if not line:
                break

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 1:
                continue

            student_id = parts[0]
            name = parts[1] if len(parts) > 1 else None
            grade = parts[2] if len(parts) > 2 else None
            class_ = parts[3] if len(parts) > 3 else None

            students.append({
                "student_id": student_id,
                "name": name,
                "grade": grade,
                "class": class_
            })

        if not students:
            print("\n❌ 未输入任何学生数据")
            return

        # 批量导入
        success = 0
        for student in students:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR IGNORE INTO students (student_id, name, grade, class)
                    VALUES (?, ?, ?, ?)
                """, (student["student_id"], student["name"],
                      student["grade"], student["class"]))

                conn.commit()
                conn.close()

                if cursor.rowcount > 0:
                    success += 1

            except Exception as e:
                print(f"\n⚠️ 导入失败 {student['student_id']}: {e}")

        print(f"\n✅ 成功导入 {success}/{len(students)} 名学生")

    def search_student(self):
        """搜索学生"""
        keyword = input("\n🔍 输入学号或姓名: ").strip()

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT student_id, name, grade, class
                FROM students
                WHERE (student_id LIKE ? OR name LIKE ?)
                AND is_active = 1
            """, (f"%{keyword}%", f"%{keyword}%"))

            results = cursor.fetchall()
            conn.close()

            if not results:
                print("\n📭 未找到匹配的学生")
                return

            print(f"\n找到 {len(results)} 名学生：\n")
            print(f"{'学号':<12} {'姓名':<10} {'年级':<8} {'班级'}")
            print("-" * 40)

            for r in results:
                print(f"{r[0]:<12} {r[1] or '-':<10} {r[2] or '-':<8} {r[3] or '-'}")

        except Exception as e:
            print(f"\n❌ 搜索失败：{e}")

    # -------------------------------------------------------------------------
    # 班级统计
    # -------------------------------------------------------------------------
    def class_statistics(self):
        """显示班级总体统计"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            print("\n" + "=" * 60)
            print("📊 班级统计")
            print("=" * 60)

            # 按班级统计
            cursor.execute("""
                SELECT grade, class,
                       COUNT(DISTINCT st.student_id) as student_count,
                       COUNT(s.session_id) as session_count,
                       AVG(s.turn_count) as avg_turns
                FROM students st
                LEFT JOIN sessions s ON st.student_id = s.student_id
                WHERE st.is_active = 1
                GROUP BY grade, class
                ORDER BY grade, class
            """)

            class_stats = cursor.fetchall()

            if not class_stats:
                print("\n📭 暂无班级数据")
                conn.close()
                return

            print(f"\n{'年级':<8} {'班级':<8} {'学生数':<8} {'会话数':<8} {'平均轮次'}")
            print("-" * 50)

            for stat in class_stats:
                grade = stat[0] or '-'
                class_ = stat[1] or '-'
                print(f"{grade:<8} {class_:<8} {stat[2]:<8} {stat[3]:<8} {stat[4]:.1f if stat[4] else 0}")

            # 风险等级分布（总体）
            cursor.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM sessions
                WHERE risk_level IS NOT NULL
                GROUP BY risk_level
            """)

            risk_dist = cursor.fetchall()

            if risk_dist:
                print(f"\n⚠️  全校风险等级分布：")
                for level, count in risk_dist:
                    print(f"  {level}: {count}人")

            conn.close()
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ 统计失败：{e}")

    # -------------------------------------------------------------------------
    # 学生评估（查看AI建议）
    # -------------------------------------------------------------------------
    def student_assessment(self):
        """查看学生评估和AI建议"""
        student_id = input("\n请输入学号: ").strip()

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 查询学生的所有会话
            cursor.execute("""
                SELECT session_id, start_time, end_time, turn_count, stage, risk_level
                FROM sessions
                WHERE student_id = ?
                ORDER BY start_time DESC
                LIMIT 10
            """, (student_id,))

            sessions = cursor.fetchall()
            conn.close()

            if not sessions:
                print(f"\n📭 学号 {student_id} 暂无会话记录")
                return

            print(f"\n学号 {student_id} 的最近会话：\n")
            print(f"{'序号':<6} {'开始时间':<19} {'轮次':<6} {'阶段':<15} {'风险'}")
            print("-" * 60)

            for idx, session in enumerate(sessions, 1):
                start_time = session[1][:19] if session[1] else "-"
                print(f"{idx:<6} {start_time:<19} {session[3] or 0:<6} {session[4] or '-':<15} {session[5] or '-'}")

            # 选择会话查看AI建议
            choice = input("\n请选择会话序号查看AI建议（输入数字，0返回）: ").strip()

            if choice == "0":
                return

            try:
                session_idx = int(choice) - 1
                if session_idx < 0 or session_idx >= len(sessions):
                    print("\n❌ 无效选择")
                    return

                session_id = sessions[session_idx][0]
                self.get_assessment_advice(session_id)

            except ValueError:
                print("\n❌ 请输入有效数字")

        except Exception as e:
            print(f"\n❌ 查询失败：{e}")

    def get_assessment_advice(self, session_id: str):
        """调用API获取AI建议"""
        try:
            print(f"\n🔄 正在生成AI建议，请稍候...")

            response = requests.get(
                f"{API_BASE}/api/teacher/assessment-advice",
                params={"session_id": session_id},
                timeout=30
            )

            if response.status_code != 200:
                print(f"\n❌ 获取建议失败：HTTP {response.status_code}")
                print(response.text)
                return

            data = response.json()

            print("\n" + "=" * 60)
            print("🤖 AI评估建议")
            print("=" * 60)
            print(f"\n学号：{data['student_id']}")
            print(f"姓名：{data['student_name']}")
            print(f"风险等级：{data['risk_level'] or '未知'}")
            print(f"对话轮次：{data['turn_count']}")

            advice = data['advice']

            print(f"\n📋 总体评估：")
            print(f"  {advice.get('overall_assessment', '暂无')}")

            if advice.get('risk_concerns'):
                print(f"\n⚠️  需要关注的风险点：")
                for concern in advice['risk_concerns']:
                    print(f"  • {concern}")

            if advice.get('teacher_advice'):
                print(f"\n💡 对老师的建议：")
                print(f"  {advice['teacher_advice']}")

            if advice.get('need_referral'):
                print(f"\n🔄 是否需要转介心理老师：{advice['need_referral']}")
                if advice.get('referral_reason'):
                    print(f"  理由：{advice['referral_reason']}")

            print("=" * 60)

        except requests.exceptions.Timeout:
            print("\n❌ 请求超时，请稍后重试")
        except Exception as e:
            print(f"\n❌ 获取建议失败：{e}")

    # -------------------------------------------------------------------------
    # 高风险对话查看（仅心理老师）
    # -------------------------------------------------------------------------
    def high_risk_conversations(self):
        """查看高风险会话对话内容（仅心理老师）"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 查询MEDIUM+HIGH风险的会话
            cursor.execute("""
                SELECT s.session_id, s.student_id, st.name, s.start_time,
                       s.turn_count, s.risk_level
                FROM sessions s
                LEFT JOIN students st ON s.student_id = st.student_id
                WHERE s.risk_level IN ('MEDIUM', 'HIGH')
                ORDER BY s.start_time DESC
                LIMIT 20
            """)

            sessions = cursor.fetchall()
            conn.close()

            if not sessions:
                print("\n📭 暂无中等或高风险会话记录")
                return

            print(f"\n找到 {len(sessions)} 个中等或高风险会话：\n")
            print(f"{'序号':<6} {'学号':<12} {'姓名':<10} {'开始时间':<19} {'风险'}")
            print("-" * 70)

            for idx, session in enumerate(sessions, 1):
                start_time = session[3][:19] if session[3] else "-"
                print(f"{idx:<6} {session[1]:<12} {session[2] or '-':<10} {start_time:<19} {session[5]}")

            choice = input("\n请选择会话序号查看对话（输入数字，0返回）: ").strip()

            if choice == "0":
                return

            try:
                session_idx = int(choice) - 1
                if session_idx < 0 or session_idx >= len(sessions):
                    print("\n❌ 无效选择")
                    return

                session_id = sessions[session_idx][0]
                self.view_conversation_detail(session_id)

            except ValueError:
                print("\n❌ 请输入有效数字")

        except Exception as e:
            print(f"\n❌ 查询失败：{e}")

    def view_conversation_detail(self, session_id: str):
        """查看会话对话详情（通过API）"""
        try:
            response = requests.get(
                f"{API_BASE}/api/teacher/session-messages",
                params={
                    "session_id": session_id,
                    "teacher_role": self.teacher_role
                },
                timeout=10
            )

            if response.status_code != 200:
                if response.status_code == 403:
                    print(f"\n❌ 权限不足：{response.json()['detail']}")
                else:
                    print(f"\n❌ 获取对话失败：HTTP {response.status_code}")
                return

            data = response.json()

            print("\n" + "=" * 60)
            print("💬 会话对话详情")
            print("=" * 60)
            print(f"\n学号：{data['student_id']}")
            print(f"姓名：{data['student_name']}")
            print(f"风险等级：{data['risk_level'] or '未知'}")
            print(f"对话记录（共 {len(data['messages'])} 条）：\n")

            for msg in data['messages']:
                role_emoji = "👤 学生" if msg['role'] == "user" else "🤖 榛子"
                timestamp = msg['timestamp'][:19] if msg['timestamp'] else ""
                print(f"[{timestamp}] {role_emoji}")
                print(f"{msg['content']}\n")

            print("=" * 60)

        except Exception as e:
            print(f"\n❌ 获取对话失败：{e}")

    def export_data(self):
        """导出数据"""
        while True:
            print("\n" + "-" * 40)
            print("📤 导出数据")
            print("-" * 40)
            print("1. 导出学生列表")
            print("2. 导出会话记录")
            print("3. 导出对话记录")
            print("0. 返回主菜单")
            print("-" * 40)

            choice = input("\n请选择操作: ").strip()

            if choice == "1":
                self.export_students()
            elif choice == "2":
                self.export_sessions()
            elif choice == "3":
                self.export_messages()
            elif choice == "0":
                break
            else:
                print("\n❌ 无效选择")

    def export_students(self):
        """导出学生列表到 CSV"""
        filename = input("\n请输入导出文件名（默认：students.csv）: ").strip() or "students.csv"

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT student_id, name, grade, class
                FROM students
                WHERE is_active = 1
                ORDER BY grade, class, student_id
            """)

            results = cursor.fetchall()
            conn.close()

            with open(filename, 'w', encoding='utf-8') as f:
                f.write("学号,姓名,年级,班级\n")
                for r in results:
                    f.write(f"{r[0]},{r[1] or ''},{r[2] or ''},{r[3] or ''}\n")

            print(f"\n✅ 成功导出到 {filename}（共 {len(results)} 条记录）")

        except Exception as e:
            print(f"\n❌ 导出失败：{e}")

    def export_sessions(self):
        """导出会话记录到 CSV"""
        filename = input("\n请输入导出文件名（默认：sessions.csv）: ").strip() or "sessions.csv"

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT s.session_id, s.student_id, st.name,
                       s.start_time, s.end_time, s.turn_count, s.stage, s.risk_level
                FROM sessions s
                LEFT JOIN students st ON s.student_id = st.student_id
                ORDER BY s.start_time DESC
            """)

            results = cursor.fetchall()
            conn.close()

            with open(filename, 'w', encoding='utf-8') as f:
                f.write("会话ID,学号,姓名,开始时间,结束时间,对话轮次,阶段,风险等级\n")
                for r in results:
                    f.write(f"{r[0]},{r[1]},{r[2] or ''},{r[3] or ''},{r[4] or ''},{r[5] or 0},{r[6] or ''},{r[7] or ''}\n")

            print(f"\n✅ 成功导出到 {filename}（共 {len(results)} 条记录）")

        except Exception as e:
            print(f"\n❌ 导出失败：{e}")

    def export_messages(self):
        """导出对话记录到 JSON"""
        filename = input("\n请输入导出文件名（默认：messages.json）: ").strip() or "messages.json"

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT s.session_id, s.student_id, st.name,
                       m.role, m.content, m.timestamp
                FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                LEFT JOIN students st ON s.student_id = st.student_id
                ORDER BY s.session_id, m.timestamp ASC
            """)

            results = cursor.fetchall()
            conn.close()

            # 按会话分组
            sessions = {}
            for r in results:
                session_id = r[0]
                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "student_id": r[1],
                        "student_name": r[2],
                        "messages": []
                    }

                sessions[session_id]["messages"].append({
                    "role": r[3],
                    "content": r[4],
                    "timestamp": r[5]
                })

            # 写入 JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(list(sessions.values()), f, ensure_ascii=False, indent=2)

            print(f"\n✅ 成功导出到 {filename}（共 {len(sessions)} 个会话）")

        except Exception as e:
            print(f"\n❌ 导出失败：{e}")


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    console = TeacherConsole()
    console.run()
# 主入口
# =============================================================================
if __name__ == "__main__":
    console = TeacherConsole()
    console.run()

# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    console = TeacherConsole()
    console.run()

