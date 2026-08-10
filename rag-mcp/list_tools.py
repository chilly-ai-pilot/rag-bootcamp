#!/usr/bin/env python3
"""
列出 RAG MCP Server 提供的 Tools

这个脚本直接导入并调用 list_tools 函数，
显示所有可用的 MCP Tools 及其参数。
"""

import sys
import os
import json
import asyncio

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(__file__))


async def main():
    """列出所有 MCP Tools"""
    # 导入 list_tools 函数（需要在函数内导入以确保路径已设置）
    from mcp_server import server
    
    print("=" * 60)
    print("RAG MCP Server - Available Tools")
    print("=" * 60)
    print()
    
    # 直接调用 list_tools 装饰的函数
    # 在 server.py 中，list_tools 函数被 @app.list_tools() 装饰
    # 我们需要找到原始函数并调用它
    tools = await server.list_tools()
    
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        print(f"   描述: {tool.description}")
        print(f"   参数:")
        
        # 显示参数信息
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            schema = tool.inputSchema
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            for prop_name, prop_info in properties.items():
                prop_type = prop_info.get('type', 'unknown')
                prop_desc = prop_info.get('description', '')
                is_required = prop_name in required
                default = prop_info.get('default', None)
                
                req_str = " (必需)" if is_required else ""
                default_str = f", 默认: {default}" if default is not None else ""
                
                print(f"      - {prop_name}: {prop_type}{req_str}{default_str}")
                if prop_desc:
                    print(f"        {prop_desc}")
        
        print()
    
    print("=" * 60)
    print(f"共 {len(tools)} 个工具")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
