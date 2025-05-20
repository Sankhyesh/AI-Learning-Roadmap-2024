// DOM Elements
const threeContainer = document.getElementById('threejs-container');
const flipBtn = document.getElementById('flip-card');
const prevBtn = document.getElementById('prev');
const nextBtn = document.getElementById('next');
const shuffleBtn = document.getElementById('shuffle');
const tagFilter = document.getElementById('tag-filter');
const cardCount = document.getElementById('card-count');
const totalCards = document.getElementById('total-cards');
const statsTotal = document.getElementById('stats-total');
const statsDue = document.getElementById('stats-due');
const statsNew = document.getElementById('stats-new');
const cardActions = document.getElementById('card-actions');

// Three.js variables
let scene, camera, renderer, cardMesh;
let frontTexture, backTexture;
const CARD_WIDTH = 8;  // Increased from 5
const CARD_HEIGHT = 5; // Increased from 3
const CARD_DEPTH = 0.15; // Slightly thicker card
const TEXTURE_WIDTH = 1536; // Higher resolution for larger card
const TEXTURE_HEIGHT = Math.round(TEXTURE_WIDTH * (CARD_HEIGHT / CARD_WIDTH));
let isAnimating = false;

// App State
let cards = [];
let filteredCards = [];
let currentCardIndex = 0;
let isShuffled = false;
let tags = new Set();

// Initialize the app
function init() {
    if (window.cardsData) {
        cards = window.cardsData;
        initThreeJS();
        setupEventListeners();
        processCards();
        updateUI();
    } else {
        console.error('No cards data available');
        threeContainer.innerHTML = '<div style="text-align: center; padding: 50px; color: #e74c3c;">Error: No flashcard data available. Please check the console for details.</div>';
    }
}

// Initialize Three.js
function initThreeJS() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f2f5);

    // Camera
    const aspect = threeContainer.clientWidth / threeContainer.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.z = 12; // Move camera back to fit larger card

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    threeContainer.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Create card
    createCard();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
}

// Create 3D card
function createCard() {
    const cardGeometry = new THREE.BoxGeometry(CARD_WIDTH, CARD_HEIGHT, CARD_DEPTH);
    
    // Create materials for each face
    const materials = [
        new THREE.MeshStandardMaterial({ color: 0xeeeeee }), // right
        new THREE.MeshStandardMaterial({ color: 0xeeeeee }), // left
        new THREE.MeshStandardMaterial({ color: 0xdddddd }), // top
        new THREE.MeshStandardMaterial({ color: 0xdddddd }), // bottom
        new THREE.MeshStandardMaterial({ color: 0xffffff }), // front (will be replaced with texture)
        new THREE.MeshStandardMaterial({ color: 0xf8f9fa })  // back (will be replaced with texture)
    ];

    cardMesh = new THREE.Mesh(cardGeometry, materials);
    scene.add(cardMesh);
    
    // Initial textures
    updateCardTextures();
}

// Update card textures based on current card
function updateCardTextures() {
    if (filteredCards.length === 0 || currentCardIndex >= filteredCards.length) {
        return;
    }

    const card = filteredCards[currentCardIndex];
    
    // Create front texture (question)
    const frontCanvas = createCardTexture(card, true);
    frontTexture = new THREE.CanvasTexture(frontCanvas);
    
    // Create back texture (answer)
    const backCanvas = createCardTexture(card, false);
    backTexture = new THREE.CanvasTexture(backCanvas);
    
    // Update materials
    cardMesh.material[4].map = frontTexture;
    cardMesh.material[4].needsUpdate = true;
    cardMesh.material[5].map = backTexture;
    cardMesh.material[5].needsUpdate = true;
    
    // Reset rotation if needed
    if (isFlipped) {
        flipCard();
    }
}

// Create canvas texture for card face
function createCardTexture(card, isFront) {
    const canvas = document.createElement('canvas');
    canvas.width = TEXTURE_WIDTH;
    canvas.height = TEXTURE_HEIGHT;
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = isFront ? '#ffffff' : '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Border
    ctx.strokeStyle = isFront ? '#3498db' : '#2ecc71';
    ctx.lineWidth = 20; // Slightly thicker border
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
    
    // Text
    const text = isFront ? 
        (card.type === 'cloze' ? card.front.replace(/\{\{c\d+::(.*?)(?:::(.*?))?\}\}/g, '[...]') : card.front) : 
        (card.cloze_back_extra || card.back);
    
    ctx.fillStyle = '#2c3e50';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Title (for front) or Answer label (for back)
    const title = isFront ? 'Question' : 'Answer';
    const titleFontSize = TEXTURE_HEIGHT / 10; // Slightly larger title
    ctx.font = `bold ${titleFontSize}px Arial`;
    ctx.fillText(title, canvas.width / 2, TEXTURE_HEIGHT / 7);
    
    // Main content
    const fontSize = TEXTURE_HEIGHT / 14; // Slightly larger font
    ctx.font = `${fontSize}px Arial`;
    
    // Wrap text
    const maxWidth = canvas.width * 0.9;
    const lineHeight = fontSize * 1.4;
    const x = canvas.width / 2;
    let y = canvas.height / 3;
    
    const lines = wrapText(ctx, text, maxWidth, fontSize);
    lines.forEach(line => {
        ctx.fillText(line, x, y);
        y += lineHeight;
    });
    
    // Add tags at the bottom for back
    if (!isFront && card.tags && card.tags.length > 0) {
        const tagsText = `Tags: ${card.tags.join(', ')}`;
        const tagsY = canvas.height * 0.88; // Move tags slightly lower
        const tagsFontSize = fontSize * 0.9; // Slightly larger tags
        ctx.font = `${tagsFontSize}px Arial`;
        ctx.fillStyle = '#7f8c8d';
        ctx.fillText(tagsText, x, tagsY);
    }
    
    return canvas;
}

// Helper to wrap text
function wrapText(context, text, maxWidth, fontSize) {
    const words = text.split(' ');
    const lines = [];
    let currentLine = words[0];
    
    for (let i = 1; i < words.length; i++) {
        const word = words[i];
        const width = context.measureText(currentLine + ' ' + word).width;
        if (width < maxWidth) {
            currentLine += ' ' + word;
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
    }
    lines.push(currentLine);
    return lines;
}

// Handle window resize
function onWindowResize() {
    const width = threeContainer.clientWidth;
    const height = threeContainer.clientHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    TWEEN.update();
    renderer.render(scene, camera);
}

// Flip card animation
function flipCard() {
    if (isAnimating) return;
    isAnimating = true;
    
    const targetRotation = isFlipped ? 0 : Math.PI;
    
    new TWEEN.Tween(cardMesh.rotation)
        .to({ y: targetRotation }, 500)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .onComplete(() => {
            isFlipped = !isFlipped;
            isAnimating = false;
            flipBtn.textContent = isFlipped ? 'Show Question' : 'Show Answer';
            
            // Show/hide card actions when showing answer
            if (cardActions) {
                cardActions.style.display = isFlipped ? 'flex' : 'none';
            }
        })
        .start();
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
        threeContainer.innerHTML = '<div style="text-align: center; padding: 50px; color: #e74c3c;">No cards match the selected filter.</div>';
        return;
    }
    
    // Reset card rotation if needed
    if (isFlipped) {
        isFlipped = false;
        cardMesh.rotation.y = 0;
        flipBtn.textContent = 'Show Answer';
        if (cardActions) {
            cardActions.style.display = 'none';
        }
    }
    
    // Update card textures
    updateCardTextures();
    
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
        shuffleBtn.textContent = ' Unshuffle';
    } else {
        // Reset to original order
        filterCards();
        shuffleBtn.textContent = ' Shuffle';
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
    // Navigation
    prevBtn.addEventListener('click', prevCard);
    nextBtn.addEventListener('click', nextCard);
    
    // Card interaction
    flipBtn.addEventListener('click', () => flipCard());
    
    // Controls
    shuffleBtn.addEventListener('click', shuffleCards);
    tagFilter.addEventListener('change', filterCards);
    
    // Rating buttons
    document.getElementById('again').addEventListener('click', () => rateCard('again'));
    document.getElementById('good').addEventListener('click', () => rateCard('good'));
    document.getElementById('easy').addEventListener('click', () => rateCard('easy'));
    
    // Hide card actions initially
    if (cardActions) {
        cardActions.style.display = 'none';
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.code === 'ArrowRight') nextCard();
        if (e.code === 'ArrowLeft') prevCard();
        if (e.code === 'Space') {
            e.preventDefault();
            flipCard();
        }
    });
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Rate the current card (simplified - in a real app, this would implement spaced repetition)
function rateCard(rating) {
    // In a real app, update the card's interval, due date, etc. based on the rating
    console.log(`Card rated: ${rating}`);
    
    // Add animation class to card
    if (cardMesh) {
        cardMesh.userData.rating = rating;
        cardMesh.scale.set(1, 1, 1);
        
        // Animate
        new TWEEN.Tween(cardMesh.scale)
            .to({ x: 1.1, y: 1.1, z: 1.1 }, 100)
            .easing(TWEEN.Easing.Quadratic.Out)
            .chain(
                new TWEEN.Tween(cardMesh.scale)
                    .to({ x: 0.8, y: 0.8, z: 0.8 }, 200)
                    .easing(TWEEN.Easing.Quadratic.In)
                    .onComplete(() => nextCard())
            )
            .start();
    } else {
        nextCard();
    }
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
