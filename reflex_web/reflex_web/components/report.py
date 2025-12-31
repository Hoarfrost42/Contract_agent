"""
报告展示组件 - Master-Detail Layout
解决无限滚动问题：左侧粘性导航 + 右侧可滚动内容
"""
import reflex as rx
from ..state import AppState
from ..styles import (
    GLASS_CARD,
    GHOST_BUTTON,
    COLORS,
    FONT_FAMILY,
)


def report_page() -> rx.Component:
    """报告页面 - Master-Detail 布局"""
    return rx.hstack(
        # ========== 左侧粘性导航侧边栏 ==========
        navigation_sidebar(),
        
        # ========== 右侧内容区域 ==========
        content_area(),
        
        spacing="0",
        width="100%",
        min_height="100vh",
    )


def navigation_sidebar() -> rx.Component:
    """左侧导航侧边栏 - 风险目录"""
    return rx.box(
        rx.vstack(
            # 返回按钮
            rx.link(
                rx.button(
                    rx.hstack(
                        rx.text("←", font_size="1.1rem"),
                        rx.text("返回工作台", font_weight="500"),
                        spacing="2",
                    ),
                    style=GHOST_BUTTON,
                    width="100%",
                    justify_content="flex-start",
                ),
                href="/",
            ),
            
            rx.divider(margin_y="16px", border_color="#E2E8F0"),
            
            # 模式切换
            rx.hstack(
                mode_toggle_button("评估总览", "summary"),
                mode_toggle_button("深度审查", "cards"),
                spacing="1",
                width="100%",
            ),
            
            rx.divider(margin_y="16px", border_color="#E2E8F0"),
            
            # 风险目录（只在深度审查模式下显示）
            rx.cond(
                AppState.report_view_mode == "cards",
                rx.vstack(
                    # 标题
                    rx.text(
                        "📋 风险目录",
                        font_size="0.85rem",
                        font_weight="700",
                        color="#64748B",
                        text_transform="uppercase",
                        letter_spacing="0.05em",
                        margin_bottom="12px",
                    ),
                    
            # 风险列表导航
                    rx.cond(
                        AppState.structured_data.length() > 0,
                        rx.vstack(
                            rx.foreach(
                                AppState.structured_data,
                                lambda item, idx: nav_item(item, idx)
                            ),
                            spacing="1",
                            width="100%",
                        ),
                        rx.text(
                            "暂无风险记录",
                            font_size="0.85rem",
                            color="#94A3B8",
                        ),
                    ),
                    spacing="2",
                    width="100%",
                    align="start",
                ),
            ),
            
            rx.spacer(),
            
            # ========== 导出功能区 ==========
            rx.divider(margin_y="16px", border_color="#E2E8F0"),
            
            rx.vstack(
                rx.text(
                    "📄 导出报告",
                    font_size="0.85rem",
                    font_weight="700",
                    color="#64748B",
                    text_transform="uppercase",
                    letter_spacing="0.05em",
                    margin_bottom="12px",
                ),
                
                # 生成 Word 按钮
                rx.button(
                    rx.hstack(
                        rx.cond(
                            AppState.word_export_loading,
                            rx.spinner(size="1"),
                            rx.text("📝", font_size="1rem"),
                        ),
                        rx.text("生成 Word 报告"),
                        spacing="2",
                        align="center",
                    ),
                    on_click=AppState.export_word_report,
                    width="100%",
                    padding="12px",
                    border_radius="10px",
                    background="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                    color="white",
                    font_weight="600",
                    border="none",
                    cursor="pointer",
                    _hover={"opacity": "0.9"},
                    disabled=AppState.word_export_loading,
                ),
                
                # 下载链接（生成后显示）
                rx.cond(
                    AppState.word_report_path != "",
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.text("⬇", font_size="1rem"),
                                rx.text("下载 Word 文件"),
                                spacing="2",
                                align="center",
                            ),
                            width="100%",
                            padding="12px",
                            border_radius="10px",
                            background="#10B981",
                            color="white",
                            font_weight="600",
                            border="none",
                            cursor="pointer",
                            _hover={"opacity": "0.9"},
                            margin_top="8px",
                        ),
                        href=AppState.word_report_path,
                        download=True,
                        is_external=True,
                    ),
                ),
                
                spacing="2",
                width="100%",
                align="start",
            ),
            
            spacing="2",
            width="100%",
            align="start",
        ),
        
        # 侧边栏样式
        width="260px",
        min_width="260px",
        height="100vh",
        padding="20px",
        background="white",
        border_right="1px solid #E2E8F0",
        overflow_y="auto",
        position="sticky",
        top="0",
        left="0",
    )


def mode_toggle_button(label: str, mode: str) -> rx.Component:
    """模式切换按钮"""
    is_active = AppState.report_view_mode == mode
    return rx.button(
        label,
        on_click=lambda: AppState.set_report_view_mode(mode),
        padding="8px 12px",
        border_radius="8px",
        font_size="0.8rem",
        font_weight="500",
        flex="1",
        background=rx.cond(is_active, "#6366F1", "transparent"),
        color=rx.cond(is_active, "white", "#64748B"),
        border="none",
        cursor="pointer",
        transition="all 0.2s ease",
        _hover={"background": rx.cond(is_active, "#6366F1", "#F1F5F9")},
    )


def nav_item(item: dict, idx: int) -> rx.Component:
    """导航项"""
    is_high_risk = item["risk_level"] == "高"
    
    return rx.link(
        rx.hstack(
            # 风险等级指示器
            rx.box(
                width="8px",
                height="8px",
                border_radius="full",
                background=rx.cond(is_high_risk, "#EF4444", "#10B981"),
            ),
            # 风险名称
            rx.text(
                f"条款 {idx + 1}",
                font_size="0.85rem",
                font_weight="500",
                color=COLORS["heading"],
                overflow="hidden",
                white_space="nowrap",
                text_overflow="ellipsis",
            ),
            spacing="3",
            width="100%",
            align="center",
        ),
        href=f"#clause-{idx}",
        width="100%",
        padding="10px 12px",
        border_radius="8px",
        border_left=rx.cond(is_high_risk, "3px solid #EF4444", "3px solid transparent"),
        background="transparent",
        _hover={"background": "#F8FAFC"},
        transition="all 0.15s ease",
    )


def content_area() -> rx.Component:
    """右侧内容区域"""
    return rx.box(
        rx.cond(
            AppState.report_view_mode == "summary",
            summary_view(),
            cards_view(),
        ),
        flex="1",
        min_height="100vh",
        padding="32px",
        background="#F8FAFC",
    )


# ==================== 模式1: 评估总览 ====================
def summary_view() -> rx.Component:
    """完整报告视图"""
    return rx.vstack(
        # Hero Score Card
        hero_score_card(),
        
        # Stats Row
        stats_row(),
        
        # Full Report Markdown
        rx.box(
            rx.text("📋 执行摘要", font_size="1.2rem", font_weight="700", color=COLORS["heading"], margin_bottom="16px"),
            rx.box(
                rx.markdown(AppState.report_md),
                background="white",
                padding="24px",
                border_radius="16px",
                border="1px solid #E2E8F0",
                font_size="0.95rem",
                line_height="1.8",
                color=COLORS["body"],
            ),
            width="100%",
            margin_top="32px",
        ),
        
        spacing="4",
        width="100%",
        max_width="900px",
    )


def hero_score_card() -> rx.Component:
    """Hero 风险评分卡片"""
    is_safe = AppState.risk_score < 30
    is_warning = (AppState.risk_score >= 30) & (AppState.risk_score < 70)
    
    return rx.box(
        rx.hstack(
            # Score Ring
            rx.box(
                rx.vstack(
                    rx.text(
                        AppState.risk_score,
                        font_size="3.5rem",
                        font_weight="800",
                        color=rx.cond(is_safe, "#059669", rx.cond(is_warning, "#D97706", "#DC2626")),
                        line_height="1",
                    ),
                    rx.text("综合风险分", font_size="0.75rem", color="#94A3B8", font_weight="600"),
                    align="center",
                    spacing="1",
                ),
                width="140px",
                height="140px",
                border_radius="50%",
                background=rx.cond(is_safe, "#ECFDF5", rx.cond(is_warning, "#FFFBEB", "#FEF2F2")),
                border=rx.cond(is_safe, "6px solid #10B981", rx.cond(is_warning, "6px solid #F59E0B", "6px solid #EF4444")),
                display="flex",
                align_items="center",
                justify_content="center",
            ),
            
            rx.spacer(),
            
            # Risk Distribution
            rx.vstack(
                rx.text("★ 风险分布", font_size="0.85rem", color="#64748B", font_weight="600"),
                rx.hstack(
                    rx.text(f"高风险: {AppState.high_risk_count}项", color="#DC2626", font_weight="600"),
                    rx.text("·", color="#CBD5E1"),
                    rx.text(f"中风险: {AppState.medium_risk_count}项", color="#F97316", font_weight="600"),
                    rx.text("·", color="#CBD5E1"),
                    rx.text(f"低风险: {AppState.low_risk_count}项", color="#059669", font_weight="600"),
                    spacing="2",
                ),
                align="start",
                spacing="2",
            ),
            
            spacing="8",
            align="center",
            width="100%",
            padding="32px",
        ),
        background="white",
        border_radius="24px",
        box_shadow="0 10px 40px rgba(0, 0, 0, 0.06)",
        width="100%",
    )


def stats_row() -> rx.Component:
    """统计指标行"""
    return rx.hstack(
        stat_item("检测条款", AppState.structured_data.length(), "处"),
        rx.divider(orientation="vertical", height="40px", border_color="#E2E8F0"),
        stat_item("高风险", AppState.high_risk_count, "处"),
        rx.divider(orientation="vertical", height="40px", border_color="#E2E8F0"),
        stat_item("中风险", AppState.medium_risk_count, "处"),
        rx.divider(orientation="vertical", height="40px", border_color="#E2E8F0"),
        stat_item("审查耗时", AppState.processing_time_formatted, "秒"),
        spacing="8",
        width="100%",
        padding="20px 0",
        justify="center",
    )


def stat_item(label: str, value, unit: str) -> rx.Component:
    return rx.vstack(
        rx.text(label, font_size="0.85rem", color="#94A3B8"),
        rx.hstack(
            rx.text(value, font_size="1.6rem", font_weight="700", color=COLORS["heading"]),
            rx.text(unit, font_size="0.85rem", color="#94A3B8", margin_left="4px"),
            align="baseline",
        ),
        align="center",
        spacing="1",
    )


# ==================== 模式2: 深度审查（卡片视图） ====================
def cards_view() -> rx.Component:
    """卡片列表视图"""
    return rx.vstack(
        # 统计信息
        rx.text(
            f"共发现 {AppState.structured_data.length()} 处风险点",
            font_size="1rem",
            font_weight="600",
            color=COLORS["heading"],
            margin_bottom="24px",
        ),
        
        # Cards List with anchors
        rx.cond(
            AppState.structured_data.length() > 0,
            rx.vstack(
                rx.foreach(
                    AppState.structured_data,
                    lambda item, idx: risk_detail_card(item, idx)
                ),
                spacing="6",
                width="100%",
            ),
            # Empty State
            rx.center(
                rx.vstack(
                    rx.text("✨", font_size="2.5rem"),
                    rx.text("暂无风险记录", font_weight="600", color=COLORS["heading"]),
                    rx.text("优秀的合同！AI 没有发现潜在风险。", color=COLORS["body"]),
                    spacing="2",
                    align="center",
                ),
                padding="60px",
                background="white",
                border_radius="20px",
                width="100%",
            )
        ),
        
        # Bottom padding for comfortable scrolling
        rx.box(height="100px"),
        
        width="100%",
        max_width="900px",
    )


def risk_detail_card(item: dict, idx: int) -> rx.Component:
    """风险详情卡片 - Split Layout（支持高/中/低三级风险）"""
    is_high_risk = item["risk_level"] == "高"
    is_medium_risk = item["risk_level"] == "中"
    
    return rx.box(
        # Card Header
        rx.box(
            rx.hstack(
                rx.text(f"条款 {idx + 1}", font_weight="700", color=COLORS["heading"]),
                rx.spacer(),
                rx.cond(
                    is_high_risk,
                    rx.hstack(
                        rx.badge("有风险隐患", variant="soft", color_scheme="orange", size="1"),
                        rx.badge("高风险", variant="solid", color_scheme="red", size="1"),
                        spacing="2",
                    ),
                    rx.cond(
                        is_medium_risk,
                        rx.hstack(
                            rx.badge("需关注", variant="soft", color_scheme="yellow", size="1"),
                            rx.badge("中风险", variant="solid", color_scheme="orange", size="1"),
                            spacing="2",
                        ),
                        rx.badge("低风险", variant="soft", color_scheme="green", size="1"),
                    ),
                ),
                width="100%",
                align="center",
            ),
            padding="16px 20px",
            border_bottom="1px solid #E2E8F0",
        ),
        
        # Card Body - Split Layout
        rx.hstack(
            # Left: 原文 + 分析
            rx.vstack(
                # 原文
                rx.box(
                    rx.text("原文全貌", font_size="0.75rem", font_weight="700", color="#6366F1", margin_bottom="8px"),
                    rx.box(
                        rx.text(item["clause_text"], font_size="0.9rem", color=COLORS["body"], line_height="1.7"),
                        background="#F8FAFC",
                        padding="12px 16px",
                        border_radius="8px",
                        border="1px solid #E2E8F0",
                    ),
                    width="100%",
                ),
                
                # 高风险：违规点 + 后果
                rx.cond(
                    is_high_risk,
                    rx.fragment(
                        rx.box(
                            rx.text("⚠ 违规点", font_size="0.75rem", font_weight="700", color="#DC2626", margin_bottom="8px"),
                            rx.text(item.get("deep_analysis", item["risk_reason"]), font_size="0.9rem", color=COLORS["body"], line_height="1.6"),
                            width="100%",
                            margin_top="16px",
                        ),
                    ),
                    rx.box(),
                ),
                
                spacing="2",
                flex="1",
                align="start",
            ),
            
            # Right: 建议 + 法律
            rx.vstack(
                # Suggestion
                rx.box(
                    rx.text("✏ 修改建议", font_size="0.75rem", font_weight="700", color="#059669", margin_bottom="8px"),
                    rx.text(
                        item.get("suggestion", "无须修改"),
                        font_size="0.9rem",
                        color=COLORS["heading"],
                        line_height="1.6",
                    ),
                    background="#F0FDF4",
                    padding="16px",
                    border_radius="12px",
                    width="100%",
                ),
                
                # Law Reference (High Risk Only)
                rx.cond(
                    is_high_risk,
                    rx.box(
                        rx.text("📚 法律依据", font_size="0.75rem", font_weight="700", color="#6366F1", margin_bottom="8px"),
                        rx.text(
                            item.get("law_content", item.get("law_reference", "无")),
                            font_size="0.85rem",
                            color=COLORS["body"],
                            line_height="1.6",
                        ),
                        background="rgba(99, 102, 241, 0.05)",
                        padding="16px",
                        border_radius="12px",
                        border="1px solid rgba(99, 102, 241, 0.1)",
                        width="100%",
                        margin_top="12px",
                    ),
                    rx.box(),
                ),
                
                width="320px",
                min_width="320px",
                align="start",
            ),
            
            spacing="6",
            width="100%",
            align="start",
            padding="20px",
        ),
        
        # Anchor ID
        id=f"clause-{idx}",
        background="white",
        border_radius="16px",
        border=rx.cond(
            is_high_risk, 
            "1px solid #FECACA",  # 高风险：红色
            rx.cond(
                is_medium_risk,
                "1px solid #FED7AA",  # 中风险：橙色
                "1px solid #E2E8F0",  # 低风险：灰色
            ),
        ),
        box_shadow="0 4px 16px rgba(0, 0, 0, 0.04)",
        width="100%",
        overflow="hidden",
        scroll_margin_top="20px",  # 锚点滚动时的顶部间距
    )
