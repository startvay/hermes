"""
自主学习引擎
基于工程控制论第14章（自适应控制）和第16章（学习系统）设计

核心功能：
- 从用户反馈中学习
- 从执行结果中提取经验
- 自动调整策略参数
- 构建知识图谱
"""

import json
import time
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """一次执行经验 v3.0+ — 支持TD-error惊喜度采样"""
    task_type: str
    action: str
    context: Dict[str, Any]
    result: bool  # 成功/失败
    feedback: Optional[str] = None
    score: float = 0.0  # 实际评分 0-10
    expected_score: float = 5.0  # 执行前预期评分 0-10（用于计算TD-error惊喜度）
    timestamp: float = field(default_factory=time.time)
    
    @property
    def td_error(self) -> float:
        """TD-error: |实际结果 - 预期结果|，越大说明越"意外"，越值得学习"""
        return abs(self.score - self.expected_score)
    
    @property
    def surprise_priority(self) -> float:
        """惊喜度优先级: 意外程度 + 结果好坏加权"""
        # 意外程度 (0-10)
        surprise = self.td_error
        # 成功经验的意外比失败经验的意外更有价值（因为成功模式值得推广）
        bonus = 2.0 if self.result else 0.0
        return surprise + bonus
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Knowledge:
    """知识条目 v3.0 — 支持树状层级"""
    domain: str  # 知识领域
    pattern: str  # 模式描述
    confidence: float  # 置信度 0-1
    evidence_count: int  # 支持证据数
    examples: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    # 树状层级支持
    hierarchy_level: int = 0  # 0=根, 1=大类, 2=具体, 3=实例
    parent_pattern: Optional[str] = None  # 父知识pattern
    children_patterns: List[str] = field(default_factory=list)  # 子知识patterns
    
    def reinforce(self, success: bool):
        """强化/弱化知识"""
        if success:
            self.confidence = min(1.0, self.confidence + 0.1 * (1 - self.confidence))
            self.evidence_count += 1
        else:
            self.confidence = max(0.0, self.confidence - 0.05)
        self.last_updated = time.time()
    
    def add_child(self, child_pattern: str):
        """添加子知识"""
        if child_pattern not in self.children_patterns:
            self.children_patterns.append(child_pattern)
    
    def is_root(self) -> bool:
        return self.hierarchy_level == 0 or self.parent_pattern is None
    
    def is_leaf(self) -> bool:
        return len(self.children_patterns) == 0


class LearningEngine:
    """
    自主学习引擎
    
    实现三大学习机制：
    1. 经验学习：从执行结果中学习
    2. 反馈学习：从用户反馈中学习
    3. 类比学习：从相似任务中迁移知识
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or Path.home() / ".hermes/evolution/data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 经验库
        self.experiences: List[Experience] = []
        
        # 知识库: domain -> pattern -> Knowledge
        self.knowledge_base: Dict[str, Dict[str, Knowledge]] = defaultdict(dict)
        
        # 策略库: task_type -> {strategy -> success_rate}
        self.strategies: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # 统计
        self.stats = {
            "total_experiences": 0,
            "total_knowledge": 0,
            "learning_cycles": 0,
            "avg_improvement": 0.0,
        }
        
        # 经验回放池（优先级 = score / 10.0）
        self.experience_buffer: List[Tuple[float, Experience]] = []
        
        # 加载已有数据
        self._load()
        
        logger.info(f"学习引擎初始化完成: {len(self.experiences)}条经验, "
                    f"{sum(len(v) for v in self.knowledge_base.values())}条知识")
    
    def record_experience(self, exp: Experience):
        """记录一次经验"""
        self.experiences.append(exp)
        self.stats["total_experiences"] += 1
        
        # 自动提取知识（成功和失败都提取，失败记录为"避免模式"）
        self._extract_knowledge(exp)
        
        # 更新策略
        self._update_strategy(exp)
        
        # 计算优先级并加入回放池（v3.0+: 使用TD-error惊喜度而非简单score）
        priority = exp.surprise_priority
        self.experience_buffer.append((priority, exp))
        
        # 保持缓冲区大小合理
        if len(self.experience_buffer) > 500:
            self.experience_buffer = sorted(self.experience_buffer, key=lambda x: x[0], reverse=True)[:400]
        
        # 定期抽象策略知识
        if len(self.experiences) % 5 == 0:
            self._abstract_strategy_knowledge()
        
        # 经验回放：定期从"最意外"的高分经验中重新学习（惊喜度优先）
        if len(self.experiences) % 5 == 0:
            self.replay_and_learn(3)
        
        # 定期知识蒸馏（每20条经验）
        if len(self.experiences) % 20 == 0:
            self._distill_meta_knowledge()
        
        # 定期保存
        if len(self.experiences) % 10 == 0:
            self._save()
    
    def replay_experiences(self, count: int = 5) -> List[Experience]:
        """按惊喜度(TD-error)回放最值得学习的经验"""
        if not self.experience_buffer:
            return []
        # v3.0+: 按惊喜度排序——越"意外"的经验越优先学习
        sorted_buffer = sorted(self.experience_buffer, key=lambda x: x[0], reverse=True)
        return [exp for _, exp in sorted_buffer[:count]]
    
    def replay_and_learn(self, count: int = 5):
        """回放高惊喜度经验并重新学习（强化记忆）"""
        top_exps = self.replay_experiences(count)
        for exp in top_exps:
            self._extract_knowledge(exp)
            self._update_strategy(exp)
        if top_exps:
            avg_surprise = sum(e.td_error for e in top_exps) / len(top_exps)
            logger.info(f"经验回放: 从{len(top_exps)}条高惊喜度经验学习 (平均TD-error={avg_surprise:.1f})")
    
    def _distill_meta_knowledge(self):
        """
        知识蒸馏：将跨domain的高置信度知识提炼为元知识
        
        原理：工程控制论Ch.14 最优滤波——从大量观测中提取本质模式
        把多个domain中相似的高置信度pattern合并成通用元知识
        """
        # 收集所有高置信度知识（confidence > 0.7）
        high_conf_knowledge = []
        for domain, patterns in self.knowledge_base.items():
            for pattern, knowledge in patterns.items():
                if knowledge.confidence >= 0.7 and knowledge.evidence_count >= 3:
                    high_conf_knowledge.append((domain, pattern, knowledge))
        
        if len(high_conf_knowledge) < 3:
            return  # 知识不够多，暂不蒸馏
        
        # 按pattern关键词聚类（简单版本：提取action关键词）
        from collections import Counter
        action_counter = Counter()
        for domain, pattern, knowledge in high_conf_knowledge:
            # 从 pattern 中提取 action（如 "use [optimize] for [task]" → optimize）
            parts = pattern.split('[')
            if len(parts) >= 2:
                action = parts[1].split(']')[0] if ']' in parts[1] else parts[1]
                action_counter[action] += 1
        
        # 找出在多个domain中出现的高频action
        meta_knowledge_created = 0
        for action, count in action_counter.most_common(5):
            if count >= 2:  # 至少在2个domain中出现
                meta_pattern = f"[通用] [{action}] 是多任务类型的可靠策略"
                meta_domain = "meta"  # 元知识域
                
                if meta_pattern not in self.knowledge_base.get(meta_domain, {}):
                    # 创建元知识
                    meta_knowledge = Knowledge(
                        domain=meta_domain,
                        pattern=meta_pattern,
                        confidence=min(0.9, 0.5 + count * 0.1),  # 出现次数越多confidence越高
                        evidence_count=count,
                        hierarchy_level=0,  # 元知识作为根
                        examples=[f"出现在{count}个domain中: {action}"],
                    )
                    if meta_domain not in self.knowledge_base:
                        self.knowledge_base[meta_domain] = {}
                    self.knowledge_base[meta_domain][meta_pattern] = meta_knowledge
                    self.stats["total_knowledge"] += 1
                    meta_knowledge_created += 1
                    logger.info(f"知识蒸馏: 提炼元知识 '{meta_pattern}' (基于{count}个domain)")
        
        if meta_knowledge_created > 0:
            logger.info(f"知识蒸馏完成: 新增{meta_knowledge_created}条元知识")
    
    def _extract_knowledge(self, exp: Experience):
        """从经验中提取知识（通用pattern: task_type -> action）"""
        domain = exp.task_type
        # 使用更通用的pattern，让同一task_type+action的经验能聚合
        pattern = f"use [{exp.action}] for [{exp.task_type}]"
        
        if pattern in self.knowledge_base[domain]:
            knowledge = self.knowledge_base[domain][pattern]
            knowledge.reinforce(exp.result)
            knowledge.examples.append(f"Score:{exp.score:.1f} Result:{exp.result}")
            # 保持examples数量合理
            if len(knowledge.examples) > 20:
                knowledge.examples = knowledge.examples[-20:]
        else:
            # 新知识：成功初始confidence 0.5，失败初始confidence 0.3
            init_conf = 0.5 if exp.result else 0.3
            knowledge = Knowledge(
                domain=domain,
                pattern=pattern,
                confidence=init_conf,
                evidence_count=1,
                examples=[f"Score:{exp.score:.1f} Result:{exp.result}"],
            )
            self.knowledge_base[domain][pattern] = knowledge
            self.stats["total_knowledge"] += 1
    
    def _abstract_strategy_knowledge(self):
        """从策略统计中抽象高层知识：哪种action对某task_type最可靠（带层级）"""
        for task_type, task_strategies in self.strategies.items():
            if not task_strategies or len(task_strategies) < 2:
                continue
            
            # 找出最佳和最差策略
            sorted_strats = sorted(task_strategies.items(), key=lambda x: x[1], reverse=True)
            best_action, best_rate = sorted_strats[0]
            worst_action, worst_rate = sorted_strats[-1]
            
            # 只有当最佳明显优于最差时才生成知识（差距>20%）
            if best_rate - worst_rate > 0.2 and best_rate > 0.5:
                domain = task_type
                
                # 1. 先生成/更新根级知识（大类）
                root_pattern = f"[{task_type}]任务最优策略"
                if root_pattern not in self.knowledge_base[domain]:
                    self.knowledge_base[domain][root_pattern] = Knowledge(
                        domain=domain,
                        pattern=root_pattern,
                        confidence=best_rate,
                        evidence_count=int(best_rate * 10),
                        hierarchy_level=0,
                        examples=[f"子策略: {best_action}({best_rate:.0%})"],
                    )
                    self.stats["total_knowledge"] += 1
                else:
                    self.knowledge_base[domain][root_pattern].confidence = best_rate
                    self.knowledge_base[domain][root_pattern].examples.append(
                        f"子策略: {best_action}({best_rate:.0%})"
                    )
                
                # 2. 生成具体策略知识（子级）
                pattern = f"优选 [{best_action}] 处理 [{task_type}] (成功率{best_rate:.0%})"
                
                if pattern not in self.knowledge_base[domain]:
                    knowledge = Knowledge(
                        domain=domain,
                        pattern=pattern,
                        confidence=best_rate,
                        evidence_count=int(best_rate * 10),
                        hierarchy_level=1,
                        parent_pattern=root_pattern,
                        examples=[f"优于 {worst_action}({worst_rate:.0%})"],
                    )
                    self.knowledge_base[domain][pattern] = knowledge
                    self.stats["total_knowledge"] += 1
                    
                    # 建立父子关系
                    self.knowledge_base[domain][root_pattern].add_child(pattern)
                    
                    logger.info(f"抽象出层级知识: {root_pattern} -> {pattern}")
    
    def get_knowledge_tree(self, domain: str) -> Dict[str, Any]:
        """获取某domain的知识树结构"""
        if domain not in self.knowledge_base:
            return {"status": "domain_not_found"}
        
        patterns = self.knowledge_base[domain]
        
        # 找根节点
        roots = [p for p, k in patterns.items() if k.is_root()]
        
        def build_tree(pattern: str, depth: int = 0) -> Dict:
            k = patterns.get(pattern)
            if not k:
                return {}
            return {
                "pattern": pattern,
                "confidence": k.confidence,
                "level": k.hierarchy_level,
                "children": [build_tree(child, depth+1) for child in k.children_patterns[:5]],
            }
        
        return {
            "domain": domain,
            "total_knowledge": len(patterns),
            "roots": [build_tree(r) for r in roots[:3]],
        }
    
    def get_all_knowledge_trees(self) -> Dict[str, Any]:
        """获取所有domain的知识树"""
        return {domain: self.get_knowledge_tree(domain) for domain in self.knowledge_base.keys()}
    
    def export_knowledge_visualization(self, output_path: str = None) -> str:
        """
        导出知识树可视化（Mermaid格式 + 文本树）
        
        Returns:
            str: Mermaid图表代码
        """
        if output_path is None:
            output_path = str(Path.home() / ".hermes/evolution/knowledge_visualization.md")
        
        lines = ["# Hermes知识树可视化\n"]
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"总知识域: {len(self.knowledge_base)}\n")
        lines.append(f"总知识条目: {sum(len(v) for v in self.knowledge_base.values())}\n\n")
        
        # Mermaid图
        lines.append("## Mermaid知识图谱\n")
        lines.append("```mermaid")
        lines.append("graph TD")
        
        node_id = 0
        node_map = {}  # pattern -> node_id
        
        for domain, patterns in self.knowledge_base.items():
            domain_node = f"D_{domain.replace(' ', '_')}"
            lines.append(f"    {domain_node}[{domain}]")
            
            for pattern, knowledge in patterns.items():
                nid = f"N{node_id}"
                node_map[(domain, pattern)] = nid
                node_id += 1
                
                label = pattern[:30] + "..." if len(pattern) > 30 else pattern
                conf = f"{knowledge.confidence:.0%}"
                lines.append(f"    {nid}[\"{label}\\n(conf:{conf})\"]")
                lines.append(f"    {domain_node} --> {nid}")
                
                # 父子连接
                if knowledge.parent_pattern:
                    parent_id = node_map.get((domain, knowledge.parent_pattern))
                    if parent_id:
                        lines.append(f"    {parent_id} --> {nid}")
        
        lines.append("```\n")
        
        # 文本树
        lines.append("## 文本知识树\n")
        for domain in sorted(self.knowledge_base.keys()):
            tree = self.get_knowledge_tree(domain)
            if tree.get('roots'):
                lines.append(f"\n### {domain}\n")
                for root in tree['roots']:
                    self._format_tree_text(root, lines, prefix="")
        
        # 保存文件
        content = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"知识可视化已导出: {output_path}")
        return content
    
    def _format_tree_text(self, node: Dict, lines: List[str], prefix: str = ""):
        """递归格式化文本树"""
        label = node.get('pattern', 'unknown')
        conf = node.get('confidence', 0)
        level = node.get('level', 0)
        indent = "  " * level
        marker = "📁" if node.get('children') else "📄"
        lines.append(f"{indent}{marker} {label} (conf:{conf:.2f})")
        for child in node.get('children', []):
            if child:
                self._format_tree_text(child, lines, prefix)
    
    def render_knowledge_graph(self, output_path: str = None) -> str:
        """
        渲染知识树为PNG图片（networkx + matplotlib）
        
        Returns:
            str: 输出文件路径
        """
        try:
            import networkx as nx
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
        except ImportError:
            logger.warning("缺少networkx或matplotlib，无法渲染知识图")
            return ""
        
        if output_path is None:
            output_path = str(Path.home() / ".hermes/evolution/knowledge_graph.png")
        
        # 构建有向图
        G = nx.DiGraph()
        
        # 颜色映射（每个domain一种颜色）
        domain_colors = [
            "#4A90D9", "#50C878", "#E74C3C", "#F39C12",
            "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
        ]
        domain_color_map = {}
        color_idx = 0
        
        node_color_map = {}
        node_size_map = {}
        
        for domain, patterns in self.knowledge_base.items():
            # 分配domain颜色
            if domain not in domain_color_map:
                domain_color_map[domain] = domain_colors[color_idx % len(domain_colors)]
                color_idx += 1
            
            # Domain节点（大节点）
            domain_node = f"[D]{domain}"
            G.add_node(domain_node, label=domain, level=-1)
            node_color_map[domain_node] = domain_color_map[domain]
            node_size_map[domain_node] = 3000
            
            for pattern, knowledge in patterns.items():
                # 截断长标签
                label = pattern[:20] + "..." if len(pattern) > 20 else pattern
                node_id = f"{domain}:{pattern}"
                G.add_node(node_id, label=label, level=knowledge.hierarchy_level)
                node_color_map[node_id] = domain_color_map[domain]
                # 根节点大一点
                node_size_map[node_id] = 1800 if knowledge.is_root() else 1200
                
                # domain -> knowledge
                G.add_edge(domain_node, node_id)
                
                # 父子关系
                if knowledge.parent_pattern:
                    parent_id = f"{domain}:{knowledge.parent_pattern}"
                    if parent_id in G:
                        G.add_edge(parent_id, node_id)
        
        # 布局：按层级分层
        pos = {}
        levels = defaultdict(list)
        for node, data in G.nodes(data=True):
            lvl = data.get('level', 0)
            levels[lvl].append(node)
        
        # 手动计算分层布局
        y_step = 2.0
        x_step = 1.5
        for lvl, nodes in sorted(levels.items()):
            y = -lvl * y_step
            count = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - count / 2) * x_step
                pos[node] = (x, y)
        
        # 如果没有pos（空图），返回
        if not pos:
            return ""
        
        # 设置中文字体
        font_prop = None
        for font_path in [
            "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
            "/mnt/c/Windows/Fonts/simsun.ttc",
        ]:
            if Path(font_path).exists():
                font_prop = fm.FontProperties(fname=font_path, size=8)
                break
        
        # 绘制
        fig, ax = plt.subplots(figsize=(16, 10))
        colors = [node_color_map.get(n, "#95A5A6") for n in G.nodes()]
        sizes = [node_size_map.get(n, 800) for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, alpha=0.5, 
                               edge_color="#7F8C8D", width=1.2, ax=ax)
        
        # 标签（使用截断后的label）
        labels = {n: d.get('label', n) for n, d in G.nodes(data=True)}
        text_items = nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="#2C3E50", ax=ax)
        if font_prop:
            for node, t in text_items.items():
                t.set_fontproperties(font_prop)
        
        ax.set_title("Hermes 知识树图谱", fontproperties=font_prop, fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"知识图谱已渲染: {output_path}")
        return output_path
    
    def _update_strategy(self, exp: Experience):
        """更新策略效果评估"""
        task_strategies = self.strategies[exp.task_type]
        strategy = exp.action
        
        if strategy not in task_strategies:
            task_strategies[strategy] = 0.5  # 初始值
        
        # 指数移动平均更新
        alpha = 0.3
        current = task_strategies[strategy]
        task_strategies[strategy] = alpha * (1.0 if exp.result else 0.0) + (1 - alpha) * current
    
    def recommend_strategy(self, task_type: str, context: Dict[str, Any] = None) -> Optional[str]:
        """
        推荐最佳策略（支持跨domain类比）
        
        Args:
            task_type: 任务类型
            context: 执行上下文
            
        Returns:
            推荐的策略名，没有则返回None
        """
        # 先尝试本domain推荐
        if task_type in self.strategies and self.strategies[task_type]:
            strategies = self.strategies[task_type]
            best = max(strategies.items(), key=lambda x: x[1])
            if best[1] > 0.5:
                logger.info(f"推荐策略: {best[0]} (成功率: {best[1]:.1%})")
                return best[0]
        
        # 本domain经验不足，尝试类比迁移
        analog_rec = self._get_analogy_recommendation(task_type)
        if analog_rec:
            return analog_rec
        
        return None
    
    def _get_analogy_recommendation(self, task_type: str) -> Optional[str]:
        """
        基于类比推理推荐策略
        
        原理: 如果其他domain中某个action成功率高，且本domain也用过这个action，
              则借鉴其他domain的评估
        """
        if not self.strategies or len(self.strategies) < 2:
            return None
        
        # 本domain已知的action
        local_actions = set(self.strategies.get(task_type, {}).keys())
        
        best_analog = None
        best_score = 0.0
        
        for other_domain, other_strategies in self.strategies.items():
            if other_domain == task_type or not other_strategies:
                continue
            
            # 找other_domain中最好的action
            best_action, best_rate = max(other_strategies.items(), key=lambda x: x[1])
            
            # 这个action在本domain也出现过 → 可以类比
            if best_action in local_actions and best_rate > best_score and best_rate > 0.5:
                best_score = best_rate
                best_analog = (other_domain, best_action, best_rate)
        
        if best_analog:
            source, action, rate = best_analog
            logger.info(f"类比推荐: {task_type} 借鉴 {source} 的 '{action}' (成功率{rate:.0%})")
            # 自动触发知识迁移
            self._transfer_analogy_knowledge(source, task_type, action, rate)
            return action
        
        # 更宽松的类比：找共享action的domain，即使本domain没评估过
        action_scores = {}
        for other_domain, other_strategies in self.strategies.items():
            if other_domain == task_type:
                continue
            for action, rate in other_strategies.items():
                if rate > 0.6:
                    if action not in action_scores or rate > action_scores[action][1]:
                        action_scores[action] = (other_domain, rate)
        
        if action_scores:
            best_action = max(action_scores.items(), key=lambda x: x[1][1])
            action, (source, rate) = best_action
            logger.info(f"类比推荐(新action): {task_type} 借鉴 {source} 的 '{action}' (成功率{rate:.0%})")
            self._transfer_analogy_knowledge(source, task_type, action, rate)
            return action
        
        return None
    
    def _transfer_analogy_knowledge(self, source_domain: str, target_domain: str, 
                                    action: str, confidence: float):
        """将source_domain的知识类比迁移到target_domain"""
        # 迁移时confidence打折（类比不如直接经验可靠）
        transferred_confidence = confidence * 0.7
        
        # 迁移策略评估
        if target_domain not in self.strategies:
            self.strategies[target_domain] = defaultdict(float)
        self.strategies[target_domain][action] = transferred_confidence
        
        # 生成迁移知识
        pattern = f"类比: {source_domain}的'{action}'可用于[{target_domain}] (置信度{transferred_confidence:.0%})"
        if pattern not in self.knowledge_base.get(target_domain, {}):
            knowledge = Knowledge(
                domain=target_domain,
                pattern=pattern,
                confidence=transferred_confidence,
                evidence_count=1,
                examples=[f"类比自 {source_domain} (原置信度{confidence:.0%})"],
            )
            if target_domain not in self.knowledge_base:
                self.knowledge_base[target_domain] = {}
            self.knowledge_base[target_domain][pattern] = knowledge
            self.stats["total_knowledge"] += 1
            logger.info(f"知识迁移: {source_domain} -> {target_domain}: {action}")
    
    def get_analogy_summary(self) -> Dict[str, Any]:
        """获取类比学习摘要"""
        analogies = []
        for domain, patterns in self.knowledge_base.items():
            for pattern, knowledge in patterns.items():
                if "类比" in pattern or "analogy" in pattern.lower():
                    analogies.append({
                        "domain": domain,
                        "pattern": pattern,
                        "confidence": knowledge.confidence,
                    })
        
        return {
            "total_analogies": len(analogies),
            "analogies": analogies[:10],
            "source_domains": list(set(self.strategies.keys())),
        }
    
    def find_similar_experiences(self, task_type: str, context: Dict[str, Any], 
                                  top_k: int = 3) -> List[Experience]:
        """找到相似的历史经验"""
        candidates = []
        
        for exp in self.experiences:
            if exp.task_type == task_type:
                # 计算上下文相似度
                similarity = self._context_similarity(context, exp.context)
                candidates.append((similarity, exp))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in candidates[:top_k]]
    
    def _context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """计算两个上下文的相似度"""
        if not ctx1 or not ctx2:
            return 0.0
        
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if ctx1[k] == ctx2[k])
        return matches / len(common_keys)
    
    def learn_from_feedback(self, task_type: str, action: str, 
                            feedback: str, score: float):
        """从用户反馈中学习"""
        exp = Experience(
            task_type=task_type,
            action=action,
            context={"feedback": feedback},
            result=score >= 5.0,
            feedback=feedback,
            score=score,
        )
        self.record_experience(exp)
        
        # 从反馈文本中提取关键词作为知识
        keywords = self._extract_keywords(feedback)
        for keyword in keywords:
            domain = f"feedback_{task_type}"
            pattern = f"keyword: {keyword}"
            
            if pattern in self.knowledge_base[domain]:
                self.knowledge_base[domain][pattern].reinforce(score >= 5.0)
            else:
                self.knowledge_base[domain][pattern] = Knowledge(
                    domain=domain,
                    pattern=pattern,
                    confidence=0.6 if score >= 5.0 else 0.3,
                    evidence_count=1,
                )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """简单关键词提取"""
        # 常见评价词
        positive = ["好", "不错", "优秀", "正确", "满意", "快", "准确", "good", "great", "excellent"]
        negative = ["差", "错", "慢", "不好", "不满意", "错误", "bad", "wrong", "slow", "poor"]
        
        keywords = []
        text_lower = text.lower()
        
        for word in positive:
            if word in text_lower:
                keywords.append(f"positive:{word}")
        for word in negative:
            if word in text_lower:
                keywords.append(f"negative:{word}")
        
        return keywords
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        self.stats["learning_cycles"] += 1
        
        # 计算各任务类型的平均成功率
        strategy_summary = {}
        for task_type, strats in self.strategies.items():
            if strats:
                best = max(strats.items(), key=lambda x: x[1])
                strategy_summary[task_type] = {
                    "best_strategy": best[0],
                    "success_rate": f"{best[1]:.1%}",
                    "total_strategies": len(strats),
                }
        
        return {
            "stats": self.stats,
            "knowledge_domains": list(self.knowledge_base.keys()),
            "strategy_summary": strategy_summary,
            "recent_experiences": len([e for e in self.experiences 
                                       if time.time() - e.timestamp < 3600]),
        }
    
    def _save(self):
        """保存学习数据"""
        try:
            # 保存经验
            exp_file = self.storage_path / "experiences.json"
            with open(exp_file, "w") as f:
                json.dump([e.to_dict() for e in self.experiences[-1000:]], f, 
                         ensure_ascii=False, indent=2)
            
            # 保存策略
            strat_file = self.storage_path / "strategies.json"
            with open(strat_file, "w") as f:
                json.dump({k: dict(v) for k, v in self.strategies.items()}, f,
                         ensure_ascii=False, indent=2)
            
            # 保存知识
            kb_file = self.storage_path / "knowledge.json"
            kb_data = {}
            for domain, patterns in self.knowledge_base.items():
                kb_data[domain] = {p: asdict(k) for p, k in patterns.items()}
            with open(kb_file, "w") as f:
                json.dump(kb_data, f, ensure_ascii=False, indent=2)
            
            logger.debug("学习数据已保存")
        except Exception as e:
            logger.error(f"保存学习数据失败: {e}")
    
    def _load(self):
        """加载已有数据"""
        try:
            # 加载经验
            exp_file = self.storage_path / "experiences.json"
            if exp_file.exists():
                with open(exp_file) as f:
                    data = json.load(f)
                    self.experiences = [Experience(**d) for d in data]
            
            # 加载策略
            strat_file = self.storage_path / "strategies.json"
            if strat_file.exists():
                with open(strat_file) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.strategies[k] = defaultdict(float, v)
            
            # 加载知识
            kb_file = self.storage_path / "knowledge.json"
            if kb_file.exists():
                with open(kb_file) as f:
                    data = json.load(f)
                    for domain, patterns in data.items():
                        for p, k_data in patterns.items():
                            self.knowledge_base[domain][p] = Knowledge(**k_data)
            
            logger.debug("学习数据已加载")
        except Exception as e:
            logger.warning(f"加载学习数据失败: {e}")
