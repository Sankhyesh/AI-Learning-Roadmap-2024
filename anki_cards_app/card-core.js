// Core functionality for 3D Anki-style flashcards
class CardCore {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            cardWidth: 8,
            cardHeight: 5,
            cardDepth: 0.15,
            textureWidth: 1536,
            isDarkMode: false,
            ...options
        };
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.cardMesh = null;
        this.isAnimating = false;
        this.isFlipped = false;
        this.currentCardIndex = 0;
        this.masterDeck = [];
        this.filteredDeck = [];
        
        this.init();
    }

    init() {
        this.initThreeJS();
    }

    initThreeJS() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.isDarkMode ? 0x1a1a1a : 0xf0f2f5);

        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(0, 1, 12);
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.setupLights();
        this.createCardMesh();
        this.animateScene();
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    setupLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
        directionalLight.position.set(5, 10, 7.5);
        this.scene.add(directionalLight);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-5, -5, -7.5);
        this.scene.add(directionalLight2);
    }

    createCardMesh() {
        const cardGeometry = new THREE.BoxGeometry(
            this.options.cardWidth,
            this.options.cardHeight,
            this.options.cardDepth
        );
        
        const sideMaterial = new THREE.MeshStandardMaterial({
            color: this.options.isDarkMode ? 0x333333 : 0xdddddd,
            roughness: 0.8,
            metalness: 0.1
        });
        
        const frontMaterial = new THREE.MeshStandardMaterial({
            map: this.createTextTexture({main: "Loading..."}, true, true),
            roughness: 0.7,
            metalness: 0.05
        });
        
        const backMaterial = new THREE.MeshStandardMaterial({
            map: this.createTextTexture({main: "Loading..."}, false, true),
            roughness: 0.7,
            metalness: 0.05
        });

        const materials = [
            sideMaterial, // right
            sideMaterial, // left
            sideMaterial, // top
            sideMaterial, // bottom
            frontMaterial, // front
            backMaterial  // back
        ];
        
        this.cardMesh = new THREE.Mesh(cardGeometry, materials);
        this.scene.add(this.cardMesh);
    }

    createTextTexture(textData, isFront, isPlaceholder = false) {
        const canvas = document.createElement('canvas');
        canvas.width = this.options.textureWidth;
        canvas.height = Math.round(this.options.textureWidth * (this.options.cardHeight / this.options.cardWidth));
        const ctx = canvas.getContext('2d');

        // Background
        ctx.fillStyle = isFront ? 
            (this.options.isDarkMode ? '#2a2a2a' : '#ffffff') : 
            (this.options.isDarkMode ? '#2d2d2d' : '#fdfdfd');
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Border
        ctx.strokeStyle = isFront ? 
            (this.options.isDarkMode ? 'rgba(52, 152, 219, 0.6)' : 'rgba(52, 152, 219, 0.8)') : 
            (this.options.isDarkMode ? 'rgba(46, 204, 113, 0.6)' : 'rgba(46, 204, 113, 0.8)');
        ctx.lineWidth = canvas.width * 0.015;
        ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, canvas.width - ctx.lineWidth, canvas.height - ctx.lineWidth);

        if (isPlaceholder) {
            ctx.fillStyle = this.options.isDarkMode ? '#666666' : '#aaaaaa';
            ctx.font = `bold ${canvas.height / 10}px 'Segoe UI', Arial, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(textData.main, canvas.width / 2, canvas.height / 2);
            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;
            return texture;
        }

        const padding = canvas.width * 0.05;
        const maxTextWidth = canvas.width - 2 * padding;
        let currentY = padding * 1.2;

        // Title
        ctx.fillStyle = isFront ? 
            (this.options.isDarkMode ? 'rgb(52, 152, 219)' : 'rgb(52, 152, 219)') : 
            (this.options.isDarkMode ? 'rgb(46, 204, 113)' : 'rgb(46, 204, 113)');
        const titleFontSize = canvas.height / 14;
        ctx.font = `bold ${titleFontSize}px 'Segoe UI', Arial, sans-serif`;
        ctx.textAlign = 'left';
        ctx.fillText(isFront ? 'Question:' : 'Answer:', padding, currentY);
        currentY += titleFontSize * 1.6;

        // Main Text
        ctx.fillStyle = this.options.isDarkMode ? 'rgb(200, 200, 200)' : 'rgb(44, 62, 80)';
        const mainFontSize = canvas.height / 18;
        ctx.font = `${mainFontSize}px 'Segoe UI', Arial, sans-serif`;
        const lines = this.wrapText(ctx, textData.main, maxTextWidth);
        lines.forEach(line => {
            if (currentY < canvas.height - padding * 2.5) {
                ctx.fillText(line, padding, currentY);
                currentY += mainFontSize * 1.35;
            }
        });

        // Tags
        if (!isFront && textData.tags && textData.tags.length > 0) {
            currentY = canvas.height - padding * 1.5;
            const tagFontSize = canvas.height / 25;
            currentY -= (tagFontSize * 1.3);

            ctx.fillStyle = this.options.isDarkMode ? '#999999' : '#7f8c8d';
            ctx.font = `italic ${tagFontSize}px 'Segoe UI', Arial, sans-serif`;
            const tagString = "Tags: " + textData.tags.join(', ');
            const tagLines = this.wrapText(ctx, tagString, maxTextWidth);
            
            for (let i = tagLines.length - 1; i >= Math.max(0, tagLines.length - 2); i--) {
                if (currentY > canvas.height - padding * 2.5) {
                    ctx.fillText(tagLines[i], padding, currentY);
                    currentY -= tagFontSize * 1.2;
                }
            }
        }

        const texture = new THREE.CanvasTexture(canvas);
        texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
        texture.needsUpdate = true;
        return texture;
    }

    wrapText(context, text, maxWidth) {
        if (!text) return [''];
        text = String(text);
        const words = text.split(' ');
        let lines = [];
        let currentLine = words[0] || '';

        for (let i = 1; i < words.length; i++) {
            const word = words[i];
            const testLine = currentLine + ' ' + word;
            const metrics = context.measureText(testLine);
            const testWidth = metrics.width;
            if (testWidth > maxWidth && currentLine !== '') {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }
        if (currentLine !== '') lines.push(currentLine);
        return lines.length > 0 ? lines : [''];
    }

    animateScene() {
        requestAnimationFrame(() => this.animateScene());
        TWEEN.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        if (!this.renderer || !this.camera || !this.container) return;
        const newWidth = this.container.clientWidth;
        const newHeight = this.container.clientHeight;
        if (newWidth === 0 || newHeight === 0) return;

        this.camera.aspect = newWidth / newHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(newWidth, newHeight);
    }

    loadDeck(cards) {
        this.masterDeck = [...cards];
        this.filteredDeck = [...cards];
        this.currentCardIndex = 0;
        this.isFlipped = false;
        this.updateCardDisplay();
    }

    updateCardDisplay() {
        if (!this.cardMesh) return;
        if (!this.filteredDeck || this.filteredDeck.length === 0) {
            this.displayNoCardsMessage();
            return;
        }

        if (this.currentCardIndex >= this.filteredDeck.length) this.currentCardIndex = 0;
        if (this.currentCardIndex < 0) this.currentCardIndex = this.filteredDeck.length - 1;

        const cardData = this.filteredDeck[this.currentCardIndex];
        let questionText = cardData.front;
        if (cardData.type === 'cloze') {
            questionText = questionText.replace(/\{\{c\d+::(.*?)(?:::(.*?))?\}\}/g, '[...]');
        }
        const answerText = cardData.cloze_back_extra || cardData.back;

        if (this.cardMesh.material[4].map) this.cardMesh.material[4].map.dispose();
        if (this.cardMesh.material[5].map) this.cardMesh.material[5].map.dispose();

        this.cardMesh.material[4].map = this.createTextTexture({ main: questionText }, true);
        this.cardMesh.material[5].map = this.createTextTexture({ main: answerText, tags: cardData.tags }, false);
        this.cardMesh.material[4].needsUpdate = true;
        this.cardMesh.material[5].needsUpdate = true;

        this.cardMesh.rotation.y = this.isFlipped ? Math.PI : 0;
    }

    displayNoCardsMessage() {
        if (this.cardMesh && this.cardMesh.material[4] && this.cardMesh.material[5]) {
            if (this.cardMesh.material[4].map) this.cardMesh.material[4].map.dispose();
            if (this.cardMesh.material[5].map) this.cardMesh.material[5].map.dispose();
            this.cardMesh.material[4].map = this.createTextTexture({main: "No cards for this filter."}, true, true);
            this.cardMesh.material[5].map = this.createTextTexture({main: "Change filter or add cards."}, false, true);
            this.cardMesh.material[4].needsUpdate = true;
            this.cardMesh.material[5].needsUpdate = true;
        }
    }

    toggleFlip() {
        if (this.isAnimating || !this.filteredDeck || this.filteredDeck.length === 0) return;
        this.isAnimating = true;

        const targetRotationY = this.isFlipped ? 0 : Math.PI;
        new TWEEN.Tween(this.cardMesh.rotation)
            .to({ y: targetRotationY }, 500)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                this.isFlipped = !this.isFlipped;
                this.isAnimating = false;
            })
            .start();
    }

    toggleDarkMode() {
        this.options.isDarkMode = !this.options.isDarkMode;
        this.scene.background = new THREE.Color(this.options.isDarkMode ? 0x1a1a1a : 0xf0f2f5);
        this.updateCardDisplay();
    }

    setupEventListeners() {
        // This should be implemented by the specific implementation
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CardCore;
} 