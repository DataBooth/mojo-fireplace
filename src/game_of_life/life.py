import time

import pygame
from gridv1 import Grid


def run_display(
    grid: Grid,
    window_height: int = 600,
    window_width: int = 600,
    background_color: str = "black",
    cell_color: str = "green",
    pause: float = 0.1,
) -> None:
    # Initialise pygame
    pygame.init()

    # Create window
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Conway's Game of Life")

    # Calculate cell size
    cell_height = window_height // grid.rows
    cell_width = window_width // grid.cols
    border_size = 1

    cell_fill_color = pygame.Color(cell_color)
    background_fill_color = pygame.Color(background_color)

    clock = pygame.time.Clock()
    running = True

    while running:
        # Event handling
        for event in pygame.event.get():  # Better than poll() for multiple events
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Clear screen
        window.fill(background_fill_color)

        # Draw live cells
        for row in range(grid.rows):
            for col in range(grid.cols):
                if grid[row, col]:
                    x = col * cell_width + border_size
                    y = row * cell_height + border_size
                    width = cell_width - 2 * border_size
                    height = cell_height - 2 * border_size

                    # Avoid drawing zero-size or negative rectangles
                    if width > 0 and height > 0:
                        pygame.draw.rect(window, cell_fill_color, (x, y, width, height))

        # Update display
        pygame.display.flip()

        # Control frame rate and pause
        time.sleep(pause)
        # Or use: clock.tick(1 / pause) for smoother timing

        # Evolve to next generation
        grid = grid.evolve()

    # Clean quit
    pygame.quit()


def main():
    start_grid = Grid.random(128, 128)
    run_display(
        start_grid,
        window_height=800,
        window_width=800,
        cell_color="lime",
        background_color="black",
        pause=0.05,  # Faster evolution
    )


if __name__ == "__main__":
    main()
