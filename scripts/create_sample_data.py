"""
创建示例数据集用于测试完整流程
生成少量虚拟论文数据以验证系统
"""

import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import ensure_dir


def create_sample_pdfs():
    """
    创建示例 PDF 文件（占位符）
    
    注意：这些是空白/简单 PDF，仅用于测试代码流程
    不能用于真实训练！需要替换为真实论文
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("错误：需要安装 reportlab")
        print("运行：pip install reportlab")
        return False
    
    print("创建示例 PDF 文件（测试用）...")
    
    # 示例数据配置
    samples = [
        # MCM A 题
        {'year': 2023, 'contest': 'MCM', 'problem': 'A', 'award': 'M', 'count': 3},
        {'year': 2023, 'contest': 'MCM', 'problem': 'A', 'award': 'H', 'count': 5},
        {'year': 2023, 'contest': 'MCM', 'problem': 'A', 'award': 'S', 'count': 2},
        
        # MCM B 题
        {'year': 2023, 'contest': 'MCM', 'problem': 'B', 'award': 'M', 'count': 3},
        {'year': 2023, 'contest': 'MCM', 'problem': 'B', 'award': 'H', 'count': 4},
        
        # ICM D 题
        {'year': 2023, 'contest': 'ICM', 'problem': 'D', 'award': 'M', 'count': 3},
        {'year': 2023, 'contest': 'ICM', 'problem': 'D', 'award': 'H', 'count': 3},
    ]
    
    total_count = 0
    
    for sample in samples:
        year = sample['year']
        contest = sample['contest']
        problem = sample['problem']
        award = sample['award']
        count = sample['count']
        
        # 创建目录
        save_dir = Path('data/raw') / str(year) / f"{contest}_{problem}"
        ensure_dir(str(save_dir))
        
        for i in range(count):
            team_id = f"{year}{ord(problem) - 65:02d}{i+1:03d}"
            filename = f"{team_id}_{award}.pdf"
            filepath = save_dir / filename
            
            # 创建简单 PDF
            c = canvas.Canvas(str(filepath), pagesize=letter)
            
            # 添加标题和摘要
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 750, f"MCM/ICM {year} Problem {problem}")
            
            c.setFont("Helvetica", 12)
            c.drawString(100, 720, f"Team {team_id}")
            c.drawString(100, 700, f"Award: {award}")
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, 660, "Abstract")
            
            c.setFont("Helvetica", 11)
            
            # 生成不同的摘要内容（根据获奖等级）
            if award == 'M':
                abstract = """
This paper presents a comprehensive mathematical model to address the problem.
We develop an innovative approach using optimization techniques and statistical
analysis. Our model achieves 95% accuracy on validation data. The results
demonstrate significant improvements over existing methods. We also conduct
sensitivity analysis to validate the robustness of our solution.
                """.strip()
            elif award == 'H':
                abstract = """
This paper proposes a model to solve the given problem. We use several
mathematical methods including linear programming and regression analysis.
The model shows reasonable performance with 80% accuracy. We discuss the
limitations and potential improvements of our approach.
                """.strip()
            else:
                abstract = """
This paper describes our approach to the problem. We apply basic mathematical
techniques to develop a simple model. The model provides acceptable results
for the given scenarios.
                """.strip()
            
            y = 640
            for line in abstract.split('\n'):
                line = line.strip()
                if line:
                    c.drawString(100, y, line)
                    y -= 15
            
            # 添加更多页面（模拟完整论文）
            for page_num in range(2, 20):
                c.showPage()
                c.setFont("Helvetica", 11)
                c.drawString(100, 750, f"Page {page_num}")
                c.drawString(100, 720, "Content of the paper...")
                
                # 每隔几页添加"图表"（实际是文字占位）
                if page_num % 3 == 0:
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(100, 680, f"Figure {page_num//3}: Sample Chart")
                    c.rect(100, 500, 400, 150, stroke=1, fill=0)
            
            c.save()
            total_count += 1
    
    print(f"\n创建完成！共生成 {total_count} 个示例 PDF")
    print("位置：data/raw/")
    print()
    print("⚠️  注意：这些是测试用的简单 PDF")
    print("   真实训练需要替换为真实的 MCM/ICM 论文！")
    print()
    return True


def verify_samples():
    """验证示例数据"""
    pdf_files = list(Path('data/raw').rglob("*.pdf"))
    
    if not pdf_files:
        print("未找到 PDF 文件")
        return False
    
    print(f"\n找到 {len(pdf_files)} 个 PDF 文件")
    print("\n按年份统计：")
    
    from collections import Counter
    years = [p.parts[-3] for p in pdf_files if len(p.parts) >= 3]
    year_counts = Counter(years)
    
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count} 篇")
    
    print("\n按题目统计：")
    problems = [p.parts[-2] for p in pdf_files if len(p.parts) >= 3]
    problem_counts = Counter(problems)
    
    for problem, count in sorted(problem_counts.items()):
        print(f"  {problem}: {count} 篇")
    
    print("\n按获奖等级统计：")
    awards = [p.stem.split('_')[-1] for p in pdf_files]
    award_counts = Counter(awards)
    
    for award, count in sorted(award_counts.items()):
        print(f"  {award}: {count} 篇")
    
    return True


def main():
    """主函数"""
    print("="*60)
    print("创建示例数据集（测试用）")
    print("="*60)
    print()
    print("这个脚本会创建约 20 个示例 PDF 文件")
    print("用于测试完整的训练流程")
    print()
    print("⚠️  警告：")
    print("   - 示例数据仅用于验证代码能否运行")
    print("   - 不能用于真实模型训练")
    print("   - 需要用真实 MCM/ICM 论文替换")
    print()
    
    confirm = input("确认创建示例数据？(y/n): ").strip().lower()
    
    if confirm != 'y':
        print("取消操作")
        return
    
    # 创建示例 PDF
    if create_sample_pdfs():
        verify_samples()
        
        print("\n" + "="*60)
        print("下一步：")
        print("="*60)
        print()
        print("测试完整流程：")
        print()
        print("  1. 预处理数据")
        print("     python scripts/prepare_data.py")
        print()
        print("  2. 训练模型")
        print("     python scripts/train.py")
        print()
        print("  3. 预测")
        print("     python predict.py --pdf data/raw/2023/MCM_A/2023000001_M.pdf --problem A --year 2023")
        print()
        print("⚠️  用真实论文替换示例数据后重新运行！")


if __name__ == "__main__":
    main()
