"""
Reflex 主应用入口
Theme: Luminous Light Theme
"""
import reflex as rx
from .state import AppState
from .components import sidebar, report_page, benchmark_page
from .styles import (
    GLOBAL_STYLE,
    LUMINOUS_BG,
    GLASS_CARD,
    GRADIENT_BUTTON,
    GHOST_BUTTON,
    UPLOAD_AREA,
    PILL_TAB_CONTAINER,
    PILL_TAB_ACTIVE,
    PILL_TAB_INACTIVE,
    COLORS,
    FONT_FAMILY,
)


def layout(content: rx.Component) -> rx.Component:
    """通用页面布局 - Luminous Background"""
    return rx.box(
        # The Luminous Canvas (Bg + Orbs)
        rx.box(style=LUMINOUS_BG),
        
        # Floating Sidebar
        sidebar(),
        
        # Main Content Area
        rx.box(
            content,
            margin_left="240px",
            width="calc(100% - 260px)",
            min_height="100vh",
            padding="40px",
            padding_top="30px",
        ),
        font_family=FONT_FAMILY,
        color=COLORS["heading"],
    )


def index() -> rx.Component:
    """首页 - Dual-Panel Layout"""
    return layout(
        rx.box(
            # Hero Section
            rx.vstack(
                rx.text(
                    "⚖️ 智能合同风险审查",
                    font_size="2.4rem",
                    font_weight="800",
                    color=COLORS["heading"],
                    font_family=FONT_FAMILY,
                    text_align="center",
                    letter_spacing="-0.03em",
                    line_height="1.2",
                ),
                rx.text(
                    "大语言模型驱动的合同合规性深度审查与风险识别系统",
                    font_size="1.1rem",
                    font_weight="400",
                    color=COLORS["body"],
                    text_align="center",
                    margin_bottom="2rem",
                ),
                align="center",
            ),
            
            # 根据状态显示不同布局
            rx.cond(
                AppState.is_loading,
                # ========== Processing Hub ==========
                rx.box(
                    processing_hub(),
                    style=GLASS_CARD,
                    width="100%",
                    max_width="560px",
                ),
                # ========== Main Dual-Panel Card ==========
                dual_panel_card(),
            ),
            
            # Center the content
            display="flex",
            flex_direction="column",
            align_items="center",
            justify_content="center",
            min_height="calc(100vh - 80px)",
            width="100%",
        )
    )


def dual_panel_card() -> rx.Component:
    """双面板主卡片 - 始终显示左右两个面板"""
    PANEL_HEIGHT = "380px"
    
    return rx.box(
        rx.vstack(
            # ========== 顶部：模式切换标签 ==========
            rx.box(
                rx.hstack(
                    rx.button(
                        "📄 上传文件",
                        on_click=lambda: AppState.set_input_method("upload"),
                        style=rx.cond(
                            AppState.input_method == "upload",
                            PILL_TAB_ACTIVE,
                            PILL_TAB_INACTIVE,
                        ),
                    ),
                    rx.button(
                        "📝 粘贴文本",
                        on_click=lambda: AppState.set_input_method("paste"),
                        style=rx.cond(
                            AppState.input_method == "paste",
                            PILL_TAB_ACTIVE,
                            PILL_TAB_INACTIVE,
                        ),
                    ),
                    spacing="1",
                ),
                style=PILL_TAB_CONTAINER,
                margin_bottom="24px",
            ),
            
            # ========== 中间：双面板分栏 ==========
            rx.hstack(
                # 左侧面板：上传或粘贴输入区
                rx.box(
                    rx.cond(
                        AppState.input_method == "upload",
                        upload_panel(),
                        paste_panel(),
                    ),
                    flex="1",
                    height=PANEL_HEIGHT,
                    background="rgba(255, 255, 255, 0.5)",
                    border_radius="20px",
                    border="1px solid rgba(226, 232, 240, 0.8)",
                    overflow="hidden",
                ),
                
                # 右侧面板：文本预览
                rx.box(
                    preview_panel(),
                    flex="1",
                    height=PANEL_HEIGHT,
                    background="rgba(255, 255, 255, 0.5)",
                    border_radius="20px",
                    border="1px solid rgba(226, 232, 240, 0.8)",
                    overflow="hidden",
                ),
                
                spacing="5",
                width="100%",
                height=PANEL_HEIGHT,
            ),
            
            # ========== 底部：设置开关 + 开始按钮 ==========
            rx.vstack(
                # 深度反思开关
                rx.hstack(
                    rx.switch(
                        checked=AppState.enable_deep_reflection,
                        on_change=AppState.set_enable_deep_reflection,
                        size="2",
                    ),
                    rx.text("深度反思模式", font_size="0.9rem", font_weight="500", color=COLORS["heading"]),
                    rx.text("(对高风险条款进行二次审查)", font_size="0.8rem", color=COLORS["body"]),
                    spacing="2",
                    align="center",
                ),
                
                # 开始按钮
                rx.button(
                    rx.hstack(
                        rx.text("🚀", font_size="1.1rem"),
                        rx.text("开始智能审查", font_weight="600"),
                        spacing="2",
                        align="center",
                        justify="center",
                    ),
                    on_click=AppState.run_analysis,
                    style=GRADIENT_BUTTON,
                    width="240px",
                ),
                
                # 提示信息
                rx.cond(
                    AppState.notification != "",
                    rx.box(
                        rx.hstack(
                            rx.text("✅", font_size="0.9rem"),
                            rx.text(AppState.notification, font_size="0.85rem", color=COLORS["body"]),
                            spacing="2",
                        ),
                        padding="10px 16px",
                        border_radius="12px",
                        background="rgba(16, 185, 129, 0.08)",
                        border="1px solid rgba(16, 185, 129, 0.2)",
                    ),
                ),
                
                spacing="4",
                align="center",
                margin_top="24px",
            ),
            
            spacing="2",
            width="100%",
            align="center",
        ),
        
        # 大玻璃卡片样式
        background="rgba(255, 255, 255, 0.7)",
        backdrop_filter="blur(24px)",
        border_radius="40px",
        border="1px solid rgba(255, 255, 255, 0.8)",
        box_shadow="0 20px 60px rgba(0, 0, 0, 0.08)",
        padding="40px",
        width="100%",
        max_width="900px",
    )


def upload_panel() -> rx.Component:
    """左侧上传面板"""
    return rx.upload(
        rx.vstack(
            rx.box(
                rx.text("📂", font_size="2.5rem"),
                width="80px",
                height="80px",
                border_radius="24px",
                background="rgba(241, 245, 249, 0.8)",
                display="flex",
                align_items="center",
                justify_content="center",
                margin_bottom="16px",
            ),
            rx.text("点击或拖拽上传", color=COLORS["heading"], font_weight="700", font_size="1.1rem"),
            rx.text("松手即开始解析", font_size="0.85rem", color=COLORS["body"], margin_top="4px"),
            rx.text("支持 PDF, DOCX, TXT", font_size="0.75rem", color="#94A3B8", margin_top="8px"),
            align="center",
            justify="center",
            spacing="1",
            height="100%",
        ),
        id="file_upload",
        on_drop=AppState.handle_upload(rx.upload_files(upload_id="file_upload")),
        width="100%",
        height="100%",
        border="2px dashed #E2E8F0",
        border_radius="20px",
        cursor="pointer",
        _hover={"border_color": "#6366F1", "background": "rgba(99, 102, 241, 0.03)"},
        transition="all 0.2s ease",
    )


def paste_panel() -> rx.Component:
    """左侧粘贴输入面板"""
    return rx.box(
        rx.text_area(
            value=AppState.contract_text,
            on_change=AppState.set_contract_text,
            placeholder="请在此粘贴需要审查的合同条款...",
            width="100%",
            height="100%",
            min_height="100%",
            border="none",
            background="transparent",
            padding="20px",
            font_size="0.9rem",
            font_family=FONT_FAMILY,
            resize="none",
            _focus={"outline": "none", "box_shadow": "none"},
        ),
        width="100%",
        height="100%",
    )


def preview_panel() -> rx.Component:
    """右侧文本预览面板"""
    return rx.cond(
        AppState.contract_text != "",
        # 有内容：显示预览
        rx.vstack(
            rx.hstack(
                rx.text("📄", font_size="1rem"),
                rx.text(
                    rx.cond(
                        AppState.uploaded_filename != "",
                        AppState.uploaded_filename,
                        "粘贴的文本"
                    ),
                    font_weight="600",
                    color=COLORS["heading"],
                    font_size="0.9rem",
                ),
                rx.spacer(),
                rx.text(f"{AppState.contract_text.length()} 字符", font_size="0.75rem", color="#94A3B8"),
                width="100%",
                align="center",
                padding="16px 20px",
                border_bottom="1px solid #E2E8F0",
            ),
            rx.scroll_area(
                rx.text(
                    AppState.contract_text,
                    font_size="0.85rem",
                    color=COLORS["body"],
                    line_height="1.7",
                    white_space="pre-wrap",
                    padding="16px 20px",
                ),
                type="always",
                scrollbars="vertical",
                style={"height": "calc(100% - 56px)"},
            ),
            spacing="0",
            width="100%",
            height="100%",
        ),
        # 无内容：显示提示
        rx.center(
            rx.vstack(
                rx.text("📋", font_size="2.5rem", opacity="0.4"),
                rx.text("文本预览", font_weight="600", color="#94A3B8", font_size="1rem"),
                rx.text("上传或粘贴合同后在此预览", font_size="0.85rem", color="#CBD5E1"),
                spacing="2",
                align="center",
            ),
            width="100%",
            height="100%",
        ),
    )


def processing_hub() -> rx.Component:
    """处理中心 - 动态智能处理动画"""
    return rx.vstack(
        # 脉动AI图标
        rx.box(
            rx.text("🧠", font_size="4rem"),
            class_name="animate-pulse",
            filter="drop-shadow(0 0 20px rgba(99, 102, 241, 0.5))",
            margin_bottom="24px",
        ),
        
        # 主状态文本
        rx.text(
            "AI 正在分析合同...",
            font_size="1.5rem",
            font_weight="700",
            color=COLORS["heading"],
            text_align="center",
        ),
        
        # 实时进度信息
        rx.text(
            AppState.notification,
            font_size="0.95rem",
            color=COLORS["body"],
            text_align="center",
            margin_top="8px",
            max_width="400px",
        ),
        
        # 进度条容器
        rx.box(
            # 进度条背景
            rx.box(
                # 流动动画填充
                rx.box(
                    width="60%",
                    height="100%",
                    background="linear-gradient(90deg, #6366F1, #8B5CF6, #6366F1)",
                    background_size="200% 100%",
                    border_radius="full",
                    class_name="animate-pulse",
                    animation="flowProgress 1.5s ease-in-out infinite",
                ),
                width="100%",
                height="8px",
                background="rgba(0,0,0,0.05)",
                border_radius="full",
                overflow="hidden",
            ),
            width="80%",
            margin_top="32px",
        ),
        
        # 提示文字
        rx.hstack(
            rx.box(
                width="8px",
                height="8px",
                background="#10B981",
                border_radius="full",
                class_name="animate-pulse",
            ),
            rx.text(
                "深度分析中，请稍候...",
                font_size="0.85rem",
                color="#64748B",
            ),
            spacing="2",
            margin_top="16px",
        ),
        
        spacing="2",
        width="100%",
        align="center",
        padding="48px 24px",
    )


def default_input_view() -> rx.Component:
    """默认输入视图 - 上传/粘贴"""
    return rx.vstack(
        # Pill Tab Switcher
        rx.box(
            rx.hstack(
                rx.button(
                    "📄 上传文件",
                    on_click=lambda: AppState.set_input_method("upload"),
                    style=rx.cond(
                        AppState.input_method == "upload",
                        PILL_TAB_ACTIVE,
                        PILL_TAB_INACTIVE,
                    ),
                ),
                rx.button(
                    "📝 粘贴文本",
                    on_click=lambda: AppState.set_input_method("paste"),
                    style=rx.cond(
                        AppState.input_method == "paste",
                        PILL_TAB_ACTIVE,
                        PILL_TAB_INACTIVE,
                    ),
                ),
                spacing="1",
            ),
            style=PILL_TAB_CONTAINER,
            margin_bottom="32px",
        ),
        
        # Content Area
        rx.cond(
            AppState.input_method == "upload",
            upload_section(),
            paste_section(),
        ),
        
        # 深度反思模式开关
        rx.hstack(
            rx.switch(
                checked=AppState.enable_deep_reflection,
                on_change=AppState.set_enable_deep_reflection,
                size="2",
            ),
            rx.text(
                "深度反思模式",
                font_size="0.9rem",
                font_weight="500",
                color=COLORS["heading"],
            ),
            rx.text(
                "(对高风险条款进行二次审查)",
                font_size="0.8rem",
                color=COLORS["body"],
            ),
            spacing="2",
            align="center",
            margin_top="20px",
        ),
        
        # Gradient Action Button
        rx.button(
            rx.hstack(
                rx.text("🚀", font_size="1.1rem"),
                rx.text("开始智能审查", font_weight="600"),
                spacing="2",
                align="center",
                justify="center",
            ),
            on_click=AppState.run_analysis,
            style=GRADIENT_BUTTON,
            margin_top="16px",
        ),
        
        # Notification (非加载时显示)
        rx.cond(
            AppState.notification != "",
            rx.box(
                rx.text(AppState.notification, font_size="0.9rem", color=COLORS["body"]),
                padding="14px 20px",
                border_radius="14px",
                background="rgba(99, 102, 241, 0.06)",
                border="1px solid rgba(99, 102, 241, 0.1)",
                margin_top="24px",
                width="100%",
            ),
        ),
        
        spacing="4",
        width="100%",
        align="center",
    )


def upload_section() -> rx.Component:
    """上传区域 - 单窗口模式（拖拽自动上传）"""
    return rx.upload(
        rx.vstack(
            rx.box(
                rx.text("📂", font_size="2.2rem"),
                width="64px",
                height="64px",
                border_radius="20px",
                background="rgba(241, 245, 249, 0.6)",
                display="flex",
                align_items="center",
                justify_content="center",
                margin_bottom="16px",
            ),
            rx.text("点击或拖拽上传合同", color=COLORS["heading"], font_weight="600", font_size="1rem"),
            rx.text("松手即开始解析，无需点击按钮", font_size="0.85rem", color=COLORS["body"], margin_top="4px"),
            align="center",
            spacing="1",
            padding="40px",
        ),
        id="file_upload",
        # 🔥 拖拽自动上传
        on_drop=AppState.handle_upload(rx.upload_files(upload_id="file_upload")),
        style=UPLOAD_AREA,
        width="100%",
    )


def paste_section() -> rx.Component:
    """粘贴区域"""
    return rx.box(
        rx.text_area(
            value=AppState.contract_text,
            on_change=AppState.set_contract_text,
            placeholder="请在此粘贴需要审查的合同条款...",
            min_height="220px",
            width="100%",
            border_radius="16px",
            background="rgba(255, 255, 255, 0.4)",
            border="1px solid rgba(255, 255, 255, 0.6)",
            padding="20px",
            font_size="0.95rem",
            font_family=FONT_FAMILY,
            color=COLORS["heading"],
            _focus={
                "border_color": COLORS["accent"],
                "background": "rgba(255, 255, 255, 0.8)",
                "box_shadow": "0 0 0 3px rgba(99, 102, 241, 0.1)",
                "outline": "none",
            },
            _placeholder={"color": "#94A3B8"},
        ),
        width="100%",
    )


def report() -> rx.Component:
    return layout(report_page())


def benchmark() -> rx.Component:
    return layout(benchmark_page())


# App Configuration
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="indigo",
        radius="large",
    ),
    style=GLOBAL_STYLE,
)

# Routes
app.add_page(index, route="/", title="Contract AI")
app.add_page(report, route="/report", title="Report")
app.add_page(benchmark, route="/benchmark", title="Benchmark")
