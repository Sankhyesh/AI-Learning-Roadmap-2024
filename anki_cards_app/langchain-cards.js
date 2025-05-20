// LangChain-specific card implementation
class LangChainCards extends CardCore {
    constructor(containerId, options = {}) {
        super(containerId, options);
        this.setupUI();
    }

    setupUI() {
        // Create UI elements
        this.createHeader();
        this.createStats();
        this.createControls();
        this.createCardActions();
        this.createCardInfo();
        
        // Get references to DOM elements
        this.initializeDOMElements();
        
        // Populate tag filter after UI is created
        this.populateTagFilter();
        
        // Initialize event listeners after all UI elements exist
        this.setupEventListeners();
        
        // Initial UI update
        this.updateUI();
    }

    initializeDOMElements() {
        // Get DOM elements
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.flipBtn = document.getElementById('flip-btn');
        this.shuffleBtn = document.getElementById('shuffle-btn');
        this.againBtn = document.getElementById('again-btn');
        this.goodBtn = document.getElementById('good-btn');
        this.easyBtn = document.getElementById('easy-btn');
        this.tagFilter = document.getElementById('tag-filter');
        this.darkModeBtn = document.getElementById('dark-mode-btn');
        this.cardActions = document.getElementById('card-actions');

        // Stats elements
        this.totalCardsStatEl = document.getElementById('total-cards');
        this.dueCardsStatEl = document.getElementById('due-cards');
        this.newCardsStatEl = document.getElementById('new-cards');
    }

    createHeader() {
        const header = document.createElement('div');
        header.className = 'header';
        header.innerHTML = `
            <h1>LangChain - 3D Flashcards</h1>
            <p>Study and review key concepts about LangChain in 3D!</p>
        `;
        this.container.parentElement.insertBefore(header, this.container);
    }

    createStats() {
        const stats = document.createElement('div');
        stats.className = 'stats';
        stats.innerHTML = `
            <div class="stat-item">
                <div class="stat-value" id="total-cards">0</div>
                <div class="stat-label">Total Cards</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="due-cards">0</div>
                <div class="stat-label">Due</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="new-cards">0</div>
                <div class="stat-label">New</div>
            </div>
        `;
        this.container.parentElement.insertBefore(stats, this.container);
    }

    createControls() {
        const controls = document.createElement('div');
        controls.className = 'controls';
        controls.innerHTML = `
            <button id="prev-btn"><i class="fas fa-arrow-left"></i> Previous</button>
            <button id="flip-btn"><i class="fas fa-sync-alt"></i> Show Answer</button>
            <button id="shuffle-btn"><i class="fas fa-random"></i> Shuffle</button>
            <select id="tag-filter">
                <option value="">All Tags</option>
            </select>
            <button id="next-btn">Next <i class="fas fa-arrow-right"></i></button>
            <button id="dark-mode-btn"><i class="fas fa-moon"></i> Dark Mode</button>
        `;
        this.container.parentElement.insertBefore(controls, this.container);
    }

    createCardActions() {
        const cardActions = document.createElement('div');
        cardActions.className = 'card-actions';
        cardActions.id = 'card-actions';
        cardActions.innerHTML = `
            <button id="again-btn"><i class="fas fa-redo"></i> Again</button>
            <button id="good-btn"><i class="fas fa-check"></i> Good</button>
            <button id="easy-btn"><i class="fas fa-star"></i> Easy</button>
        `;
        this.container.parentElement.insertBefore(cardActions, this.container);
    }

    createCardInfo() {
        const cardInfo = document.createElement('div');
        cardInfo.className = 'card-info';
        cardInfo.innerHTML = `
            <h2>How to use these flashcards:</h2>
            <ul>
                <li>Click <strong>Show Answer</strong> to flip the card and reveal the answer.</li>
                <li>Use the <strong>Previous</strong> and <strong>Next</strong> buttons to navigate between cards.</li>
                <li>Keyboard shortcuts: <strong>Space</strong> to flip, <strong>→</strong> for next, <strong>←</strong> for previous.</li>
                <li>Rate each card after answering to help with spaced repetition learning.</li>
                <li>Use the tag filter to focus on specific topics.</li>
                <li>Toggle dark mode for comfortable viewing in low-light conditions.</li>
            </ul>
        `;
        this.container.parentElement.appendChild(cardInfo);
    }

    setupEventListeners() {
        // Add event listeners
        this.prevBtn.addEventListener('click', () => this.showPrevCard());
        this.nextBtn.addEventListener('click', () => this.showNextCard());
        this.flipBtn.addEventListener('click', () => this.toggleFlip());
        this.shuffleBtn.addEventListener('click', () => this.shuffleAndFilterCards());
        this.tagFilter.addEventListener('change', () => this.shuffleAndFilterCards());
        this.darkModeBtn.addEventListener('click', () => this.toggleDarkMode());

        this.againBtn.addEventListener('click', () => this.handleCardRated('again'));
        this.goodBtn.addEventListener('click', () => this.handleCardRated('good'));
        this.easyBtn.addEventListener('click', () => this.handleCardRated('easy'));

        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
    }

    populateTagFilter() {
        const allTags = new Set();
        this.masterDeck.forEach(card => {
            if (card.tags && Array.isArray(card.tags)) {
                card.tags.forEach(tag => allTags.add(tag));
            }
        });
        this.tagFilter.innerHTML = '<option value="">All Tags</option>';
        Array.from(allTags).sort().forEach(tag => {
            const option = document.createElement('option');
            option.value = tag;
            option.textContent = tag;
            this.tagFilter.appendChild(option);
        });
    }

    shuffleAndFilterCards() {
        const selectedTag = this.tagFilter.value;
        let tempDeck = [...this.masterDeck];

        if (selectedTag) {
            tempDeck = tempDeck.filter(card => card.tags && card.tags.includes(selectedTag));
        }
        this.shuffleArray(tempDeck);
        this.filteredDeck = tempDeck;
        this.currentCardIndex = 0;
        this.isFlipped = false;
        if (this.cardMesh) this.cardMesh.rotation.y = 0;
        this.updateCardDisplay();
        this.updateUI();
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    showNextCard() {
        if (this.currentCardIndex < this.filteredDeck.length - 1) {
            this.currentCardIndex++;
            this.isFlipped = false;
            if(this.cardMesh) this.cardMesh.position.set(0,0,0);
            this.updateCardDisplay();
            this.updateUI();
        }
    }

    showPrevCard() {
        if (this.currentCardIndex > 0) {
            this.currentCardIndex--;
            this.isFlipped = false;
            if(this.cardMesh) this.cardMesh.position.set(0,0,0);
            this.updateCardDisplay();
            this.updateUI();
        }
    }

    handleCardRated(rating) {
        console.log(`Card ID: ${this.filteredDeck[this.currentCardIndex]?.id || 'N/A'} rated: ${rating}`);
        if (this.currentCardIndex < this.filteredDeck.length - 1) {
            this.showNextCard();
        } else {
            alert("End of filtered deck reached! Reshuffling or change filter.");
            this.shuffleAndFilterCards();
        }
    }

    handleKeyDown(e) {
        if (this.isAnimating) return;
        
        const noCardsAvailable = !this.filteredDeck || this.filteredDeck.length === 0;

        switch(e.code) {
            case 'ArrowRight':
                if (!this.nextBtn.disabled && !noCardsAvailable) this.showNextCard();
                break;
            case 'ArrowLeft':
                if (!this.prevBtn.disabled && !noCardsAvailable) this.showPrevCard();
                break;
            case 'Space':
                e.preventDefault();
                if (!this.flipBtn.disabled && !noCardsAvailable) this.toggleFlip();
                break;
            case 'KeyS':
                if (!this.shuffleBtn.disabled) this.shuffleBtn.click();
                break;
            case 'Digit1':
                if (this.isFlipped && !this.againBtn.disabled && !noCardsAvailable) this.againBtn.click();
                break;
            case 'Digit2':
                if (this.isFlipped && !this.goodBtn.disabled && !noCardsAvailable) this.goodBtn.click();
                break;
            case 'Digit3':
                if (this.isFlipped && !this.easyBtn.disabled && !noCardsAvailable) this.easyBtn.click();
                break;
            case 'KeyD':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    this.toggleDarkMode();
                }
                break;
        }
    }

    updateUI() {
        const hasCards = this.filteredDeck && this.filteredDeck.length > 0;
        this.totalCardsStatEl.textContent = this.masterDeck.length;
        this.dueCardsStatEl.textContent = 'N/A';
        this.newCardsStatEl.textContent = hasCards ? this.filteredDeck.length : '0';

        this.flipBtn.innerHTML = `<i class="fas fa-sync-alt"></i> ${this.isFlipped ? 'Show Question' : 'Show Answer'}`;
        this.cardActions.style.display = this.isFlipped && hasCards ? 'flex' : 'none';

        this.prevBtn.disabled = !hasCards || this.currentCardIndex === 0;
        this.nextBtn.disabled = !hasCards || this.currentCardIndex === this.filteredDeck.length - 1;
        this.flipBtn.disabled = !hasCards;
        this.shuffleBtn.disabled = this.masterDeck.length < 2;
        this.tagFilter.disabled = this.masterDeck.length === 0;

        [this.againBtn, this.goodBtn, this.easyBtn].forEach(btn => 
            btn.disabled = !hasCards || !this.isFlipped
        );

        // Update dark mode button
        this.darkModeBtn.innerHTML = `<i class="fas fa-${this.options.isDarkMode ? 'sun' : 'moon'}"></i> ${this.options.isDarkMode ? 'Light' : 'Dark'} Mode`;
    }

    toggleDarkMode() {
        super.toggleDarkMode();
        document.body.classList.toggle('dark-mode');
        this.updateUI();
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LangChainCards;
} 