"""
play.py  --  play Rock-Paper-Scissors against the learning AI in the terminal.
This is your live demo for the presentation.

Run it with:   python play.py
Type r, p or s to play. Type q to quit and see your stats.
"""

from game import NAMES, result
from predictors import FrequencyPredictor, MarkovPredictor


def choose_ai():
    print("Choose the AI:")
    print("  1) Frequency  (learns your favourite move)")
    print("  2) Markov     (learns what you play after each move)")
    while True:
        pick = input("Enter 1 or 2: ").strip()
        if pick == "1":
            return FrequencyPredictor()
        if pick == "2":
            return MarkovPredictor()


def main():
    ai = choose_ai()
    print(f"\nPlaying against the {ai.name} AI. Type r / p / s, or q to quit.\n")

    wins = draws = losses = 0   # from the AI's point of view
    rounds = 0
    history = []                # 1 if AI won that round, else 0

    while True:
        key = input("Your move (r/p/s, q to quit): ").strip().upper()
        if key == "Q":
            break
        if key not in NAMES:
            print("Please type r, p or s.")
            continue

        guess = ai.predict()                 # what the AI thinks you will play
        ai_move = ai.choose_move()
        r = result(ai_move, key)
        rounds += 1
        history.append(1 if r == 1 else 0)

        if r == 1:
            wins += 1
            outcome = "AI wins"
        elif r == 0:
            draws += 1
            outcome = "Draw"
        else:
            losses += 1
            outcome = "You win"

        guess_text = NAMES[guess] if guess else "no idea yet"
        print(f"  AI guessed: {guess_text:<12} AI played: {NAMES[ai_move]:<9} -> {outcome}")
        print(f"  Score  AI {wins} | Draws {draws} | You {losses}   "
              f"AI win rate: {100 * wins / rounds:.0f}%\n")

        ai.update(key)                       # the AI learns from your move

    if rounds:
        print(f"\nGame over. {rounds} rounds.  AI wins {wins}, draws {draws}, you won {losses}.")
        print(f"Overall AI win rate: {100 * wins / rounds:.1f}%  (chance level is 33.3%)")
        if rounds >= 20:
            half = rounds // 2
            first = 100 * sum(history[:half]) / half
            second = 100 * sum(history[half:]) / (rounds - half)
            print(f"AI win rate, first half: {first:.0f}%   second half: {second:.0f}%")


if __name__ == "__main__":
    main()
