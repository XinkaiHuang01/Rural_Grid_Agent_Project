import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# =====================================================================
# 智慧农电微型电网调度与评估 Agent 系统 (Multi-Agent Workflow)
# =====================================================================

class RuralGridAgentSystem:
    def __init__(self):
        """初始化 Agent 系统，配置大模型客户端"""
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        self.model = "gpt-4o" # 可根据实际情况替换，如 gpt-3.5-turbo 等

    def _call_agent(self, system_prompt, user_input, response_format=None):
        """Agent 通用调用引擎"""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.2, # 保持推理的客观性和稳定性
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # ---------------------------------------------------------
    # Agent 1: 结构化解析 Agent (Data Parsing Agent)
    # ---------------------------------------------------------
    def agent_data_parser(self, raw_text):
        """提取调研报告中的非结构化文本，转化为结构化 JSON 数据"""
        print("[Agent 1] 正在清洗非结构化调研数据...")
        sys_prompt = """你是一个专业的电力数据清洗Agent。
你的任务是从实地调研的口语化、非结构化文本中，提取关键电力参数。
必须以严格的 JSON 格式输出，包含以下字段：
- location: 地点
- peak_load_time: 用电高峰时段
- main_load_source: 主要用电负荷来源
- pv_potential_sqm: 闲置屋顶面积(平方米)
- current_pv_rate: 当前光伏接入率
"""
        result = self._call_agent(sys_prompt, raw_text, response_format="json")
        return json.loads(result)

    # ---------------------------------------------------------
    # Agent 2: 策略推理 Agent (Strategy Reasoning Agent)
    # ---------------------------------------------------------
    def agent_strategy_planner(self, structured_data):
        """基于长链推理，分析电网特征并生成调度策略"""
        print("[Agent 2] 正在进行长链推理与调度策略生成...")
        data_str = json.dumps(structured_data, ensure_ascii=False)
        sys_prompt = """你是一个高级微电网调度决策Agent。
请根据提供的结构化乡村电网数据，进行逻辑推理并输出专业的电网改造策略。
包含：1. 负荷特征分析；2. 分布式光伏接入建议；3. 削峰填谷调度策略。
请直接输出策略内容，专业严谨。"""
        
        result = self._call_agent(sys_prompt, f"电网数据：\n{data_str}")
        return result

    # ---------------------------------------------------------
    # Agent 3: 报告生成 Agent (Report Generation Agent)
    # ---------------------------------------------------------
    def agent_report_writer(self, structured_data, strategy):
        """整合所有信息，输出 Markdown 格式的可视化评估报告"""
        print("[Agent 3] 正在整合最终方案报告...")
        sys_prompt = "你是一个项目报告撰写Agent。负责将前期的数据和策略汇总为格式清晰的Markdown报告。"
        user_prompt = f"""
请基于以下信息生成《乡村智慧微电网改造可行性评估报告》：
【基础数据】：{json.dumps(structured_data, ensure_ascii=False)}
【调度策略】：{strategy}

要求：包含标题、项目背景、数据分析、核心调度算法建议、预期效益五个部分。使用 Markdown 排版。
"""
        result = self._call_agent(sys_prompt, user_prompt)
        return result

    # ---------------------------------------------------------
    # 主流程 Pipeline
    # ---------------------------------------------------------
    def run_pipeline(self, raw_survey_text):
        """执行多 Agent 协同流"""
        print(">>> 启动智慧农电 Agent 协同流 <<<\n")
        
        parsed_data = self.agent_data_parser(raw_survey_text)
        print(f"✅ 数据解析完成: {parsed_data}\n")
        
        strategy = self.agent_strategy_planner(parsed_data)
        print("✅ 策略推理完成\n")
        
        final_report = self.agent_report_writer(parsed_data, strategy)
        print("✅ 报告生成完成\n")
        
        return final_report

if __name__ == "__main__":
    system = RuralGridAgentSystem()
    
    # 模拟实地调研获得的非结构化杂乱数据
    mock_survey_data = """
     【实地调研数据输入模板】
     调研区域：[某特定下沉市场/乡村节点]
     用电痛点：在 [特定高负载时段，如夏季午后]，由于 [特定大功率设备，如排灌机械] 集中运行，导致配网末端电压偏低及频繁跳闸。
     新能源潜力：经勘测，区域内可用闲置屋顶面积约 [具体数值，如 5000] 平方米，当前光伏渗透率为 [低数值]%。
     硬件现状：配电变压器等基础设施老化。
     需求：生成针对该节点的智能微网改造与柔性调度策略。
    """
    
    report = system.run_pipeline(mock_survey_data)
    
    print("="*40)
    print("最终输出成果：\n")
    print(report)
    
    with open("Rural_Grid_Report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n报告已保存至 Rural_Grid_Report.md")
