"""
Glass Sidebar - 浮动毛玻璃导航面板
Apple Glassmorphism Design
"""
import reflex as rx
from ..state import AppState
from ..styles import (
    GLASS_SIDEBAR, 
    NAV_ITEM_ACTIVE, 
    NAV_ITEM_INACTIVE, 
    COLORS, 
    FONT_FAMILY,
)


def sidebar() -> rx.Component:
    """渲染浮动毛玻璃侧边栏"""
    return rx.box(
        rx.vstack(
            # Logo/Brand
            rx.hstack(
                rx.box(
                    rx.text("⚖️", font_size="1.3rem"),
                    width="40px",
                    height="40px",
                    border_radius="12px",
                    background="linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                ),
                rx.text(
                    "Contract AI", 
                    font_weight="700", 
                    font_size="1.1rem", 
                    color=COLORS["heading"],
                    font_family=FONT_FAMILY,
                ),
                spacing="3",
                align="center",
            ),
            
            rx.box(height="24px"),  # Spacer
            
            # Navigation Links
            nav_link("🏠", "审查", "/"),
            nav_link("📊", "报告", "/report"),
            nav_link("🧪", "评测", "/benchmark"),
            
            rx.spacer(),
            
            # Status Indicator
            rx.box(
                rx.hstack(
                    rx.box(
                        width="8px",
                        height="8px",
                        border_radius="50%",
                        background="#22C55E",  # green-500
                    ),
                    rx.text("System Online", font_size="0.8rem", color=COLORS["muted"]),
                    spacing="2",
                    align="center",
                ),
                padding="12px 16px",
                background="rgba(34, 197, 94, 0.08)",
                border_radius="12px",
            ),
            
            height="100%",
            width="100%",
            padding="24px",
            align="start",
        ),
        # Floating Glass Panel Style
        style=GLASS_SIDEBAR,
        width="200px",
        height="calc(100vh - 40px)",  # Leave margin from edges
        position="fixed",
        left="20px",  # Float from edge
        top="20px",
        z_index="100",
    )


def nav_link(icon: str, label: str, href: str) -> rx.Component:
    """导航链接"""
    is_active = rx.State.router.page.path == href
    
    return rx.link(
        rx.hstack(
            rx.text(icon, font_size="1rem"),
            rx.text(
                label, 
                font_size="0.95rem", 
                font_weight=rx.cond(is_active, "600", "500"),
                color=rx.cond(is_active, COLORS["accent"], COLORS["body"]),
                font_family=FONT_FAMILY,
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        href=href,
        width="100%",
        text_decoration="none",
        padding="14px 16px",
        border_radius="14px",
        background=rx.cond(is_active, "rgba(99, 102, 241, 0.12)", "transparent"),
        _hover={
            "background": "rgba(0, 0, 0, 0.03)",
            "text_decoration": "none",
        },
        transition="all 0.2s ease",
    )
