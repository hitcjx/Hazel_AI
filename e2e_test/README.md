# E2E 测试目录

本目录包含端到端自动化测试脚本，用于测试系统的四阶段完整咨询流程。

## 文件说明

- `run_test.py` - 主测试脚本

## 运行测试

```bash
cd /home/dazzle/Hazel_AI/e2e_test
python run_test.py
```

## 输出

- `report.html` - HTML格式的测试报告

## 清理

如需删除此测试模块，只需删除整个 `e2e_test` 目录即可，不会影响主项目：

```bash
rm -rf /home/dazzle/Hazel_AI/e2e_test
```
