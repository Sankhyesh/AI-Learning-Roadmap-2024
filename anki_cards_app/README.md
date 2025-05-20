# AI Learning Flashcards

A simple, interactive flashcard application for learning AI and machine learning concepts, inspired by Anki.

## Features

- **Interactive Flashcards**: Flip cards to reveal answers
- **Tag Filtering**: Filter cards by topic/tag
- **Shuffle**: Randomize the order of cards
- **Responsive Design**: Works on desktop and mobile devices
- **Keyboard Navigation**: Use arrow keys to navigate between cards
- **Basic Spaced Repetition**: Rate cards as "Again", "Good", or "Easy"

## Getting Started

1. Clone or download this repository
2. Open `index.html` in your web browser
3. Start learning!

## Adding New Cards

1. Edit the `sample-cards.json` file to add or modify flashcards
2. Each card should follow this format:
   ```json
   {
     "front": "Question or term",
     "back": "Answer or definition",
     "type": "basic",  // or "cloze" for cloze deletion cards
     "tags": ["tag1", "tag2"],
     "id": "unique-id"
   }
   ```

## Keyboard Shortcuts

- **Right Arrow** or **Space**: Next card
- **Left Arrow**: Previous card
- **Up/Down Arrows**: Flip the current card

## Future Improvements

- User authentication
- Cloud sync for progress
- More advanced spaced repetition algorithm
- Import/export functionality
- Dark mode

## License

This project is open source and available under the [MIT License](LICENSE).
