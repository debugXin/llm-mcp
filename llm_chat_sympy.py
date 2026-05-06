#!/usr/bin/env python3
"""
LLM Chat Application with SymPy MCP Integration
支持多模型配置，对话中自动调用 SymPy 进行数学计算，混合推理模式

核心特性：
- LLM 在对话中可以自主决定何时调用数学工具
- 工具调用结果会返回给 LLM，LLM 基于精确计算结果继续推理
- 用户看到的是完整的、连贯的回复，工具调用过程完全透明
"""

import json
import re
import os
import sys
from typing import Optional, List, Dict, Any
from sympy import sympify, N, Symbol, symbols, integrate, diff, solve, simplify, expand, factor
from sympy.core.numbers import Float, Rational

# 尝试导入 httpx 用于 API 调用
try:
    import httpx
except ImportError:
    print("正在安装 httpx...")
    os.system("pip install httpx -q")
    import httpx


class SymPyMCP:
    """SymPy 数学计算引擎 - 模拟 MCP 工具"""
    
    @staticmethod
    def calculate(expression: str) -> str:
        """
        计算数学表达式
        :param expression: 数学表达式字符串，如 "4*2", "sin(pi/2)", "integrate(x**2, x)"
        :return: 计算结果字符串
        """
        try:
            # 预处理表达式
            expr_str = expression.strip()
            
            # 替换常见的数学符号
            replacements = {
                'π': 'pi',
                '×': '*',
                '÷': '/',
                '^': '**',
                '√': 'sqrt',
            }
            for old, new in replacements.items():
                expr_str = expr_str.replace(old, new)
            
            # 解析并计算
            result = sympify(expr_str, evaluate=True)
            
            # 格式化输出
            if isinstance(result, (Float, Rational)):
                # 数值结果
                return str(N(result, 15))  # 15 位精度
            elif result.is_number:
                return str(N(result, 15))
            else:
                # 符号结果
                return str(result)
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def solve_equation(equation: str, variable: str = 'x') -> str:
        """解方程"""
        try:
            eq = sympify(equation)
            var = Symbol(variable)
            solutions = solve(eq, var)
            return str(solutions)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def differentiate(expression: str, variable: str = 'x') -> str:
        """求导"""
        try:
            expr = sympify(expression)
            var = Symbol(variable)
            result = diff(expr, var)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def integrate_expr(expression: str, variable: str = 'x') -> str:
        """积分"""
        try:
            expr = sympify(expression)
            var = Symbol(variable)
            result = integrate(expr, var)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def simplify_expr(expression: str) -> str:
        """化简"""
        try:
            expr = sympify(expression)
            result = simplify(expr)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class LLMChatWithMath:
    """支持数学工具调用的 LLM 聊天类
    
    工作流程：
    1. 用户发送消息
    2. LLM 分析是否需要调用数学工具
    3. 如果需要，LLM 返回工具调用请求
    4. 系统执行工具调用，获取精确计算结果
    5. 将结果返回给 LLM
    6. LLM 基于计算结果继续推理，生成最终回复
    7. 用户看到完整的、包含精确计算的回复
    """
    
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self.sympy_mcp = SymPyMCP()
        self.max_tool_iterations = 5  # 最大工具调用次数，防止无限循环
    
    def _build_tool_definition(self) -> List[Dict[str, Any]]:
        """构建工具定义，告诉 LLM 可以调用哪些数学工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "sympy_calculate",
                    "description": "执行精确的数学计算。当你需要计算数学表达式、验证计算结果、或举例说明时调用此工具。例如：解释乘法时计算 4*2，讲解微积分时求导或积分。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "要计算的数学表达式，使用标准数学符号，如 '4*2', '123*456', 'sin(pi/2)', 'x**2 + 3*x - 5'"
                            },
                            "operation": {
                                "type": "string",
                                "description": "操作类型",
                                "enum": ["calculate", "solve", "differentiate", "integrate", "simplify"]
                            },
                            "variable": {
                                "type": "string",
                                "description": "变量名（用于微积分或方程求解），默认为 'x'",
                                "default": "x"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def _call_llm_api(self, messages: List[Dict[str, str]], use_tools: bool = False) -> Dict[str, Any]:
        """调用 LLM API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        if use_tools:
            payload["tools"] = self._build_tool_definition()
            payload["tool_choice"] = "auto"
        
        try:
            response = httpx.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            error_msg = f"API 调用失败：{str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\n响应：{e.response.text}"
            raise Exception(error_msg)
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """执行工具调用"""
        function = tool_call.get('function', {})
        func_name = function.get('name', '')
        arguments = function.get('arguments', '{}')
        
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            # 尝试修复可能的 JSON 格式问题
            arguments = arguments.replace("'", '"').replace('True', 'true').replace('False', 'false')
            args = json.loads(arguments)
        
        expression = args.get('expression', '')
        operation = args.get('operation', 'calculate')
        variable = args.get('variable', 'x')
        
        print(f"\n🔧 [SymPy MCP] 调用：{operation}('{expression}')")
        
        if operation == 'calculate':
            result = self.sympy_mcp.calculate(expression)
        elif operation == 'solve':
            result = self.sympy_mcp.solve_equation(expression, variable)
        elif operation == 'differentiate':
            result = self.sympy_mcp.differentiate(expression, variable)
        elif operation == 'integrate':
            result = self.sympy_mcp.integrate_expr(expression, variable)
        elif operation == 'simplify':
            result = self.sympy_mcp.simplify_expr(expression)
        else:
            result = self.sympy_mcp.calculate(expression)
        
        print(f"✅ [SymPy MCP] 结果：{result}")
        return result
    
    def chat(self, user_message: str) -> str:
        """
        与 LLM 对话，自动处理数学工具调用
        
        这是核心方法，实现混合推理模式：
        - LLM 负责理解问题、组织语言、决定何时需要计算
        - SymPy 负责提供精确的计算结果
        - LLM 基于精确结果继续推理，生成最终回复
        
        :param user_message: 用户消息
        :return: LLM 回复
        """
        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 限制历史记录长度
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        iteration = 0
        final_response = ""
        
        while iteration < self.max_tool_iterations:
            iteration += 1
            
            # 调用 LLM（启用工具调用功能）
            response_data = self._call_llm_api(
                self.conversation_history,
                use_tools=True
            )
            
            choice = response_data.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            # 检查是否有工具调用
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                # LLM 决定调用工具（例如需要计算 4*2）
                print(f"\n📞 LLM 请求 {len(tool_calls)} 个工具调用")
                
                # 将 LLM 的回复（包含工具调用请求）添加到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.get('content', ''),
                    "tool_calls": tool_calls
                })
                
                # 执行每个工具调用
                for tool_call in tool_calls:
                    tool_id = tool_call.get('id', '')
                    result = self._execute_tool(tool_call)
                    
                    # 将工具结果返回给 LLM
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result
                    })
                
                # 继续下一轮，让 LLM 基于工具结果生成最终回复
                # 这就是"混合推理"的关键：LLM 拿到精确计算结果后继续推理
                continue
            else:
                # 没有工具调用，生成最终回复
                final_response = message.get('content', '')
                
                # 添加助手回复到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                break
        
        if iteration >= self.max_tool_iterations:
            print("\n⚠️ 达到最大工具调用次数限制")
        
        return final_response
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("✅ 对话历史已清空")


def load_config(config_file: str = "llm_config.json") -> Dict[str, str]:
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
            return configs
    return {}


def save_config(configs: Dict[str, str], config_file: str = "llm_config.json"):
    """保存配置文件"""
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)


def configure_models():
    """配置 LLM 模型"""
    configs = load_config()
    
    print("\n=== 配置 LLM 模型 ===")
    print("当前已配置的模型:")
    if configs:
        for name, config in configs.items():
            print(f"  - {name}: {config['model_name']} ({config['api_url']})")
    else:
        print("  (无)")
    
    print("\n选项:")
    print("  1. 添加新模型")
    print("  2. 删除模型")
    print("  3. 返回主菜单")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '1':
        name = input("模型名称 (如：gpt-4, qwen-max): ").strip()
        if not name:
            print("❌ 模型名称不能为空")
            return
        
        api_url = input("API URL (如：https://api.openai.com/v1): ").strip()
        api_key = input("API Key: ").strip()
        model_name = input("Model Name (如：gpt-4o): ").strip()
        
        if not all([api_url, api_key, model_name]):
            print("❌ 所有字段都不能为空")
            return
        
        configs[name] = {
            "api_url": api_url,
            "api_key": api_key,
            "model_name": model_name
        }
        save_config(configs)
        print(f"✅ 模型 '{name}' 已添加")
    
    elif choice == '2':
        if not configs:
            print("❌ 没有可删除的模型")
            return
        
        name = input("要删除的模型名称：").strip()
        if name in configs:
            del configs[name]
            save_config(configs)
            print(f"✅ 模型 '{name}' 已删除")
        else:
            print(f"❌ 模型 '{name}' 不存在")


def test_sympy():
    """测试 SymPy 功能"""
    print("\n=== 测试 SymPy 计算引擎 ===")
    
    test_cases = [
        "4*2",
        "123*456",
        "sin(pi/2)",
        "cos(pi)",
        "sqrt(16)",
        "2**10",
        "(3+4)*5",
    ]
    
    mcp = SymPyMCP()
    
    for expr in test_cases:
        result = mcp.calculate(expr)
        print(f"  {expr} = {result}")
    
    print("\n✅ SymPy 测试完成")


def main():
    """主程序入口"""
    print("=" * 70)
    print("  LLM Chat with SymPy MCP - 混合推理模式")
    print("  LLM 负责推理和解释，SymPy 负责精确计算")
    print("  示例：让模型讲解乘法，它会在举例时自动调用 SymPy 计算 4*2")
    print("=" * 70)
    
    current_model = None
    chat_session = None
    
    while True:
        print("\n=== 主菜单 ===")
        if current_model:
            print(f"当前模型：{current_model}")
        else:
            print("当前模型：(未选择)")
        
        print("1. 配置模型")
        print("2. 选择模型")
        print("3. 开始聊天")
        print("4. 测试 SymPy")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            configure_models()
        
        elif choice == '2':
            configs = load_config()
            if not configs:
                print("❌ 请先配置模型")
                continue
            
            print("\n可用模型:")
            for i, (name, _) in enumerate(configs.items(), 1):
                print(f"  {i}. {name}")
            
            try:
                idx = int(input("选择模型编号：").strip())
                model_names = list(configs.keys())
                if 1 <= idx <= len(model_names):
                    current_model = model_names[idx - 1]
                    config = configs[current_model]
                    chat_session = LLMChatWithMath(
                        api_url=config['api_url'],
                        api_key=config['api_key'],
                        model_name=config['model_name']
                    )
                    print(f"✅ 已选择模型：{current_model}")
                else:
                    print("❌ 无效的选择")
            except ValueError:
                print("❌ 请输入有效的数字")
        
        elif choice == '3':
            if not chat_session:
                print("❌ 请先选择模型")
                continue
            
            print("\n=== 聊天模式 ===")
            print("输入 'clear' 清空历史，'quit' 退出聊天")
            print("💡 试试：'请给我讲解一下乘法，并举几个例子'\n")
            
            while True:
                user_input = input("你：").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'clear':
                    chat_session.clear_history()
                    continue
                elif not user_input:
                    continue
                
                try:
                    response = chat_session.chat(user_input)
                    print(f"\n助手：{response}\n")
                except Exception as e:
                    print(f"\n❌ 错误：{e}\n")
        
        elif choice == '4':
            test_sympy()
        
        elif choice == '5':
            print("\n👋 再见！")
            break
        
        else:
            print("❌ 无效的选择")


if __name__ == "__main__":
    main()
