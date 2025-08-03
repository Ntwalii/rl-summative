import pygame
import os

# Constants
WIDTH, HEIGHT = 400, 600
MSG_WIDTH = 320
MSG_HEIGHT = 40
AGENT_Y = HEIGHT - 80

# Color codes
COLORS = {
    0: (100, 200, 255),   # Good message (blue)
    1: (255, 165, 0),     # Suspicious (orange)
    2: (255, 0, 0),       # Scam (red)
    "background": (30, 30, 30),
    "agent": (0, 255, 0),  # Agent green box
}

pygame.init()
font = pygame.font.SysFont(None, 24)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Anti-Scam Message Flow")

def draw_message(obs, action_taken, step, record=False):
    screen.fill(COLORS["background"])

    # Draw falling message
    msg_type = int(obs[2])
    urgency = obs[1]
    sender_rep = obs[0]
    color = COLORS[msg_type]

    y_pos = int((step % 10) * (MSG_HEIGHT + 5))
    msg_rect = pygame.Rect(40, y_pos, MSG_WIDTH, MSG_HEIGHT)
    pygame.draw.rect(screen, color, msg_rect)

    label = font.render(f"Reputation: {sender_rep:.2f}, Urgency: {urgency:.2f}", True, (0, 0, 0))
    screen.blit(label, (msg_rect.x + 5, msg_rect.y + 8))

    # Draw agent decision
    action_text = ["Allow", "Block", "Investigate"][action_taken]
    pygame.draw.rect(screen, COLORS["agent"], (40, AGENT_Y, MSG_WIDTH, 50))
    action_label = font.render(f"Agent: {action_text}", True, (0, 0, 0))
    screen.blit(action_label, (60, AGENT_Y + 15))

    pygame.display.flip()
    pygame.time.wait(1000)

    # Save frame if recording
    if record:
        os.makedirs("gif_frames", exist_ok=True)
        frame_path = f"gif_frames/frame_{step:03d}.png"
        pygame.image.save(screen, frame_path)
