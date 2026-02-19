class GameEngine:
    def __init__(self):
        self.xp = 0
        self.streak = 0
        self.level = 1

    def update_score(self, score):
        if score > 75:
            self.streak += 1
            self.xp += 5
        else:
            self.streak = 0

        self.level = self.xp // 200 + 1
