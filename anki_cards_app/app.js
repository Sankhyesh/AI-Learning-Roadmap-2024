// DOM Elements
const card = document.getElementById('card');
const frontContent = document.getElementById('front-content');
const backContent = document.getElementById('back-content');
const prevBtn = document.getElementById('prev');
const nextBtn = document.getElementById('next');
const shuffleBtn = document.getElementById('shuffle');
const tagFilter = document.getElementById('tag-filter');
const cardCount = document.getElementById('card-count');
const totalCards = document.getElementById('total-cards');
const statsTotal = document.getElementById('stats-total');
const statsDue = document.getElementById('stats-due');
const statsNew = document.getElementById('stats-new');

// App State
let cards = [];
let filteredCards = [];
let currentCardIndex = 0;
let isShuffled = false;
let tags = new Set();

// Initialize the app
function init() {
    // Cards are now loaded by the inline script in index.html
    // and stored in window.cardsData
    if (window.cardsData) {
        cards = window.cardsData;
        setupEventListeners();
        updateUI();
    } else {
        console.error('No cards data available');
        frontContent.textContent = 'Error: No flashcard data available. Please check the console for details.';
    }
}

// Extract tags from cards and populate the tag filter
function processCards() {
    // Clear existing tags
    tags.clear();
    
    // Extract all unique tags
    cards.forEach(card => {
        if (card.tags && Array.isArray(card.tags)) {
            card.tags.forEach(tag => tags.add(tag));
        }
    });
    
    // Clear existing tag filter options
    tagFilter.innerHTML = '<option value="">All Tags</option>';
    
    // Populate tag filter
    populateTagFilter();
    
    // Initialize filtered cards
    filterCards();
    
    // Update UI
    updateStats();
    showCurrentCard();
}

// Populate the tag filter dropdown
function populateTagFilter() {
    const sortedTags = Array.from(tags).sort();
    
    sortedTags.forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = tag;
        tagFilter.appendChild(option);
    });
}

// Filter cards based on selected tag
function filterCards() {
    const selectedTag = tagFilter.value;
    
    if (!selectedTag) {
        filteredCards = [...cards];
    } else {
        filteredCards = cards.filter(card => 
            card.tags && card.tags.includes(selectedTag)
        );
    }
    
    // Reset current card index
    currentCardIndex = 0;
    
    // Update UI
    updateUI();
}

// Show the current card
function showCurrentCard() {
    if (filteredCards.length === 0) {
        frontContent.innerHTML = 'No cards match the selected filter.';
        backContent.innerHTML = '';
        card.classList.remove('flipped');
        return;
    }
    
    const currentCard = filteredCards[currentCardIndex];
    
    // Display front content (handle cloze deletions if present)
    let frontText = currentCard.front;
    if (currentCard.type === 'cloze') {
        // Simple cloze handling - remove {{c1::...}} and show blanks
        frontText = frontText.replace(/\{\{c\d+::(.*?)(?:::(.*?))?\}\}/g, '[...]');
    }
    frontContent.innerHTML = frontText;
    
    // Display back content
    let backText = currentCard.back;
    if (currentCard.type === 'cloze' && currentCard.cloze_back_extra) {
        backText = currentCard.cloze_back_extra;
    }
    backContent.innerHTML = backText;
    
    // Update card counter
    cardCount.textContent = currentCardIndex + 1;
    totalCards.textContent = filteredCards.length;
    
    // Reset card flip state
    card.classList.remove('flipped');
}

// Navigate to the next card
function nextCard() {
    if (filteredCards.length === 0) return;
    
    currentCardIndex = (currentCardIndex + 1) % filteredCards.length;
    showCurrentCard();
}

// Navigate to the previous card
function prevCard() {
    if (filteredCards.length === 0) return;
    
    currentCardIndex = (currentCardIndex - 1 + filteredCards.length) % filteredCards.length;
    showCurrentCard();
}

// Shuffle the cards
function shuffleCards() {
    if (filteredCards.length === 0) return;
    
    isShuffled = !isShuffled;
    
    if (isShuffled) {
        // Fisher-Yates shuffle algorithm
        for (let i = filteredCards.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [filteredCards[i], filteredCards[j]] = [filteredCards[j], filteredCards[i]];
        }
        shuffleBtn.textContent = '🔀 Unshuffle';
    } else {
        // Reset to original order
        filterCards();
        shuffleBtn.textContent = '🔀 Shuffle';
    }
    
    currentCardIndex = 0;
    showCurrentCard();
}

// Update the stats display
function updateStats() {
    statsTotal.textContent = cards.length;
    statsDue.textContent = cards.length; // Simplified - in a real app, this would track due cards
    statsNew.textContent = cards.length; // Simplified - in a real app, this would track new cards
}

// Update the UI based on current state
function updateUI() {
    // Update navigation buttons
    prevBtn.disabled = filteredCards.length === 0 || filteredCards.length === 1;
    nextBtn.disabled = filteredCards.length === 0 || filteredCards.length === 1;
    shuffleBtn.disabled = filteredCards.length === 0;
    
    // Update card counter
    cardCount.textContent = filteredCards.length === 0 ? '0' : currentCardIndex + 1;
    totalCards.textContent = filteredCards.length;
    
    // Show current card
    showCurrentCard();
}

// Set up event listeners
function setupEventListeners() {
    // Card flip
    card.addEventListener('click', () => {
        card.classList.toggle('flipped');
    });
    
    // Navigation
    prevBtn.addEventListener('click', prevCard);
    nextBtn.addEventListener('click', nextCard);
    
    // Shuffle
    shuffleBtn.addEventListener('click', shuffleCards);
    
    // Tag filter
    tagFilter.addEventListener('change', filterCards);
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.code === 'ArrowRight' || e.code === 'Space') {
            nextCard();
        } else if (e.code === 'ArrowLeft') {
            prevCard();
        } else if (e.code === 'ArrowUp' || e.code === 'ArrowDown') {
            card.classList.toggle('flipped');
        }
    });
    
    // Card rating buttons
    document.getElementById('again').addEventListener('click', () => rateCard('again'));
    document.getElementById('good').addEventListener('click', () => rateCard('good'));
    document.getElementById('easy').addEventListener('click', () => rateCard('easy'));
}

// Rate the current card (simplified - in a real app, this would implement spaced repetition)
function rateCard(rating) {
    // In a real app, this would update the card's scheduling
    console.log(`Card rated as: ${rating}`);
    
    // For now, just move to the next card
    nextCard();
}

// Expose the app functions to the window object
window.app = {
    init: function() {
        processCards();
        init();
    },
    loadNewCards: function(newCards) {
        if (Array.isArray(newCards)) {
            cards = newCards;
            currentCardIndex = 0;
            processCards();
            updateUI();
            return true;
        }
        return false;
    }
};

// Initialize the app when the DOM is loaded
// Note: The actual initialization is now handled by the inline script in index.html
// to ensure cards are loaded first
document.addEventListener('DOMContentLoaded', function() {
    // The app will be initialized after cards are loaded
});
