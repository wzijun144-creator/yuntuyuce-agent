import os
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from mobilenet3diaoyong5 import FusionInferenceEngine

# 1. 启动推理引擎
engine = FusionInferenceEngine()

# 2. 将 predict_latest 包装成 Agent 工具
def get_satellite_warning_status(query=""):
    res = engine.predict_latest()
    if res['status'] == 'success':
        return (
            f"分析完成。目标时间 {res['prediction_target_time']}，"
            f"当前云量 {res['current_cloud_coverage']}，"
            f"预测云量 {res['predicted_cloud_coverage']}，"
            f"预测光伏功率 {res['predicted_power_kw']} kW，"
            f"功率波动率 {res['ramp_rate']:.1%}，"
            f"突变概率 {res['mutation_probability']}。"
            f"功率警报：{'⚠️ 是' if res['power_alert'] else '否'}。"
            f"最终决策：{res['alert_decision']}。"
            f"分析依据：{res['detailed_analysis']}"
        )
    else:
        return f"工具运行异常：{res['message']}"

# 3. 注册工具箱
tools = [
    Tool(
        name="Weather_Mutation_Monitor",
        func=get_satellite_warning_status,
        description="当用户询问天气风险、云量突变、光伏功率波动或光伏预警时，"
                    "必须使用此工具获取实时风云卫星数据及光伏功率预测结果。"
                    "调用后，你会得到一份完整的分析报告，请基于报告内容直接回答用户。"
    )
]

# 4. 配置大脑（LLM）
os.environ["OPENAI_API_KEY"] = "sk-30853b265f1f47e7b765a31568b196a9"
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url="https://api.deepseek.com"
)

# 5. 初始化 Agent（同时传入控制参数）
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=3,                 # 限制最多 3 次工具调用
    handle_parsing_errors=True        # 允许自动修复解析错误
)

if __name__ == "__main__":
    print("=== 气象 Agent 启动成功 ===")
    response = agent.invoke({"input": "你好，请帮我看看现在的卫星云图，未来一段时间有光伏波动的风险吗？"})
    print(f"\nAgent 最终回答：{response['output']}")