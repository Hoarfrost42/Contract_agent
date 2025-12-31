"""
Benchmark 评测组件 - Luminous Light Theme
Clean, Simple, Functional
"""
import reflex as rx
from ..state import AppState
from ..styles import (
    GLASS_CARD,
    GLASS_CARD_SUBTLE,
    GRADIENT_BUTTON,
    COLORS,
    FONT_FAMILY,
)


def benchmark_page() -> rx.Component:
    """Benchmark 评测页面"""
    return rx.box(
        # Header
        rx.hstack(
            rx.box(
                rx.text("🧪", font_size="1.5rem"),
                width="48px",
                height="48px",
                border_radius="14px",
                background="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                display="flex",
                align_items="center",
                justify_content="center",
            ),
            rx.vstack(
                rx.text("消融实验控制台", font_size="1.4rem", font_weight="700", color=COLORS["heading"]),
                rx.text("Laboratory Benchmark System", font_size="0.85rem", color=COLORS["muted"]),
                align="start",
                spacing="0",
            ),
            spacing="4",
            align="center",
            margin_bottom="32px",
        ),
        
        # Main Control Panel Card
        rx.box(
            rx.vstack(
                # Section 1: Mode Selection
                rx.box(
                    rx.hstack(
                        rx.text("🎯", font_size="1.1rem"),
                        rx.text("评测模式", font_weight="600", color=COLORS["heading"]),
                        spacing="2",
                        align="center",
                        margin_bottom="16px",
                    ),
                    rx.hstack(
                        pill_toggle(1, "模式 1", "纯LLM"),
                        pill_toggle(2, "模式 2", "基础Prompt"),
                        pill_toggle(3, "模式 3", "工作流"),
                        pill_toggle(4, "模式 4", "优化CoT"),
                        spacing="3",
                        flex_wrap="wrap",
                    ),
                    margin_bottom="28px",
                ),
                
                # Section 2: Configuration
                rx.box(
                    rx.hstack(
                        rx.text("⚙️", font_size="1.1rem"),
                        rx.text("实验配置", font_weight="600", color=COLORS["heading"]),
                        spacing="2",
                        align="center",
                        margin_bottom="16px",
                    ),
                    rx.hstack(
                        # Data Path - 改为下拉菜单
                        rx.box(
                            rx.text("测试数据集", font_size="0.8rem", color=COLORS["muted"], margin_bottom="8px"),
                            rx.select(
                                AppState.available_datasets,
                                value=AppState.ablation_data_path,
                                on_change=AppState.set_ablation_path,
                            ),
                            flex="2",
                        ),
                        # Sample Count
                        rx.box(
                            rx.text("样本数量", font_size="0.8rem", color=COLORS["muted"], margin_bottom="8px"),
                            rx.input(
                                value=AppState.ablation_limit,
                                on_change=AppState.set_ablation_limit,
                                type="number",
                                placeholder="5",
                                width="100%",
                                height="44px",
                                border_radius="12px",
                                background="white",
                                border="1px solid #E2E8F0",
                                padding="0 16px",
                                color=COLORS["heading"],
                                _focus={
                                    "border_color": COLORS["accent"],
                                    "box_shadow": "0 0 0 3px rgba(99, 102, 241, 0.1)",
                                },
                            ),
                            flex="1",
                        ),
                        # LLM Source
                        rx.box(
                            rx.text("LLM 源", font_size="0.8rem", color=COLORS["muted"], margin_bottom="8px"),
                            rx.select(
                                ["local", "cloud"],
                                value=AppState.ablation_source,
                                on_change=AppState.set_ablation_source,
                            ),
                            flex="1",
                        ),
                        spacing="4",
                        width="100%",
                    ),
                    margin_bottom="28px",
                ),
                
                # Divider
                rx.box(
                    height="1px",
                    background="linear-gradient(90deg, transparent, #E2E8F0, transparent)",
                    width="100%",
                    margin_y="8px",
                ),
                
                # Bottom Action Button
                rx.button(
                    rx.hstack(
                        rx.cond(
                            AppState.ablation_running,
                            rx.spinner(size="2", color="white"),
                            rx.text("🚀", font_size="1.2rem"),
                        ),
                        rx.text("启动消融实验", font_weight="700", color="white"),
                        spacing="3",
                        align="center",
                        justify="center",
                    ),
                    on_click=AppState.run_ablation,
                    disabled=AppState.ablation_running,
                    style=GRADIENT_BUTTON,
                    margin_top="20px",
                ),
                
                spacing="4",
                width="100%",
            ),
            style=GLASS_CARD,
            width="100%",
        ),
        
        # Notification
        rx.cond(
            AppState.notification != "",
            rx.box(
                rx.hstack(
                    rx.text("📢", font_size="1rem"),
                    rx.text(AppState.notification, font_size="0.9rem", color=COLORS["body"]),
                    spacing="2",
                    align="center",
                ),
                padding="16px 20px",
                border_radius="14px",
                background="rgba(99, 102, 241, 0.08)",
                border="1px solid rgba(99, 102, 241, 0.1)",
                margin_top="20px",
            ),
        ),
        
        # ==================== 图表展示区域 ====================
        rx.cond(
            AppState.ablation_combined_chart != "",
            rx.box(
                rx.vstack(
                    # Section Header
                    rx.hstack(
                        rx.text("📊", font_size="1.1rem"),
                        rx.text("评测图表", font_weight="600", color=COLORS["heading"]),
                        spacing="2",
                        align="center",
                        margin_bottom="16px",
                    ),
                    
                    # Combined Chart
                    rx.box(
                        rx.text("综合指标对比", font_weight="600", font_size="0.9rem", color=COLORS["heading"], margin_bottom="12px"),
                        rx.image(
                            src=AppState.ablation_combined_chart,
                            width="100%",
                            border_radius="12px",
                            box_shadow="0 2px 8px rgba(0,0,0,0.08)",
                        ),
                        margin_bottom="24px",
                    ),
                    
                    # Individual Charts
                    rx.text("各指标详细", font_weight="600", font_size="0.9rem", color=COLORS["heading"], margin_bottom="12px"),
                    rx.flex(
                        rx.foreach(
                            AppState.ablation_chart_paths,
                            lambda path: rx.box(
                                rx.image(
                                    src=path,
                                    width="100%",
                                    border_radius="10px",
                                    box_shadow="0 2px 6px rgba(0,0,0,0.06)",
                                ),
                                width="48%",
                                margin_bottom="16px",
                            ),
                        ),
                        flex_wrap="wrap",
                        justify="between",
                        width="100%",
                    ),
                    
                    width="100%",
                ),
                style=GLASS_CARD,
                width="100%",
                margin_top="24px",
            ),
        ),
        
        width="100%",
        max_width="800px",
        margin="0 auto",
    )


def pill_toggle(mode: int, title: str, subtitle: str) -> rx.Component:
    """Pill Toggle 组件"""
    is_active = AppState.ablation_modes.contains(mode)
    
    return rx.box(
        rx.vstack(
            rx.text(title, font_weight="600", font_size="0.9rem", color=rx.cond(is_active, "#1D4ED8", COLORS["heading"])),
            rx.text(subtitle, font_size="0.75rem", color=rx.cond(is_active, "#3B82F6", COLORS["muted"])),
            align="center",
            spacing="0",
        ),
        padding="12px 20px",
        border_radius="9999px",
        background=rx.cond(is_active, "rgba(219, 234, 254, 0.8)", "rgba(248, 250, 252, 0.6)"),
        border=rx.cond(is_active, "1px solid rgba(191, 219, 254, 0.8)", "1px solid transparent"),
        cursor="pointer",
        box_shadow=rx.cond(is_active, "0 0 12px rgba(59, 130, 246, 0.2)", "none"),
        transition="all 0.2s ease",
        on_click=lambda: AppState.toggle_ablation_mode(mode),
        _hover={
            "background": rx.cond(is_active, "rgba(219, 234, 254, 0.9)", "rgba(241, 245, 249, 0.8)"),
        },
    )
