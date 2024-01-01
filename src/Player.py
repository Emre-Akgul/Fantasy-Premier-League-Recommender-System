class Player:
    def __init__(self, player_id: int):
        self.id = player_id

    def __repr__(self) -> str:
        return f"Player(id={self.id})"