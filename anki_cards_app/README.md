# 3D AI Learning Flashcards

An interactive 3D flashcard application for learning AI and machine learning concepts, with smooth animations and an engaging user interface. Built with Three.js for stunning 3D card effects.

## Features

- **3D Flashcards**: Beautiful 3D cards with smooth flip animations
- **Interactive UI**: Intuitive controls and visual feedback
- **Tag Filtering**: Filter cards by topic/tag
- **Shuffle**: Randomize the order of cards with a single click
- **Responsive Design**: Works on desktop and mobile devices
- **Keyboard Navigation**: Use arrow keys to navigate and space to flip cards
- **Card Rating**: Rate cards as "Again", "Good", or "Easy" to track your learning progress
- **Visual Feedback**: Smooth animations for card flips and transitions

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

- **→ (Right Arrow)**: Next card
- **← (Left Arrow)**: Previous card
- **Space**: Flip the current card
- **Enter**: Flip the current card (alternative to space)

## Technical Details

- Built with Three.js for 3D rendering
- Uses Tween.js for smooth animations
- Responsive design that works on all screen sizes
- Clean, modern UI with intuitive controls

## Future Improvements

- User authentication and cloud sync
- More advanced spaced repetition algorithm
- Import/export functionality
- Dark mode
- 3D card customization options
- Progress tracking and statistics
- Support for images and rich media in cards

## License

This project is open source and available under the [MIT License](LICENSE).
