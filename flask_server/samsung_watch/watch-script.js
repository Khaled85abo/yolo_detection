// WebSocket connection
let socket;
let connectionStatus = 'disconnected';
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 1000;
let currentScreen = 0;
const screens = ['status-screen', 'controls-screen', 'rules-screen'];
let rules = {};

// Initialize the app
window.addEventListener('load', function () {
    connectWebSocket();
    setupSwipeNavigation();
});

// Save rules button event
document.querySelector('.btn-save-rules').addEventListener('click', saveRules);

// Setup swipe navigation
function setupSwipeNavigation() {
    let touchStartX = 0;
    let touchEndX = 0;

    document.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    });

    document.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });

    function handleSwipe() {
        const swipeThreshold = 50;

        if (touchEndX < touchStartX - swipeThreshold) {
            // Swipe left
            navigateToNextScreen();
        }

        if (touchEndX > touchStartX + swipeThreshold) {
            // Swipe right
            navigateToPrevScreen();
        }
    }
}

function navigateToNextScreen() {
    if (currentScreen < screens.length - 1) {
        currentScreen++;
        updateScreenPositions();
    }
}

function navigateToPrevScreen() {
    if (currentScreen > 0) {
        currentScreen--;
        updateScreenPositions();
    }
}

function updateScreenPositions() {
    screens.forEach((screenId, index) => {
        const screen = document.getElementById(screenId);
        screen.style.transform = `translateX(${(index - currentScreen) * 100}%)`;
    });
}

function connectWebSocket() {
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = protocol + window.location.host + '/ws';

    console.log('Connecting to WebSocket at:', wsUrl);

    // Close existing socket if it exists
    if (socket) {
        try {
            socket.close();
        } catch (e) {
            console.error("Error closing existing socket:", e);
        }
    }

    socket = new WebSocket(wsUrl);

    socket.onopen = function () {
        console.log('Connected to server websocket');
        connectionStatus = 'connected';
        document.getElementById('connection-status').textContent = 'Connected';
        document.querySelector('.status-indicator').classList.remove('status-initializing', 'status-disconnected');
        document.querySelector('.status-indicator').classList.add('status-connected');
        reconnectAttempts = 0;
    };

    socket.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);

            // Handle different event types
            if (data.event === 'status_update') {
                updateStatusDisplay(data.data);
            }
            if (data.event === 'rules_update') {
                updateRulesDisplay(data.data);
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
        document.getElementById('connection-status').textContent = 'Error';
        document.querySelector('.status-indicator').classList.remove('status-connected', 'status-initializing');
        document.querySelector('.status-indicator').classList.add('status-disconnected');
    };

    socket.onclose = function (event) {
        connectionStatus = 'disconnected';
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.querySelector('.status-indicator').classList.remove('status-connected', 'status-initializing');
        document.querySelector('.status-indicator').classList.add('status-disconnected');

        // Handle reconnection
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = reconnectDelay * Math.min(reconnectAttempts, 5);
            setTimeout(connectWebSocket, delay);
        } else {
            document.getElementById('connection-status').textContent = 'Failed';
        }
    };
}

function updateStatusDisplay(data) {
    document.getElementById('status').innerHTML = `
        <p>Conveyor: <strong>${data.conveyor_stop ? '⛔ STOP' : '✅ RUN'}</strong></p>
        <p>Overlap: <strong>${data.overlap ? '⚠️ YES' : '✅ NO'}</strong></p>
        <p>Stopped: <strong>${data.stop ? '⚠️ YES' : '✅ NO'}</strong></p>
        <p>Incorrect: <strong>${data.incorrect ? '⚠️ YES' : '✅ NO'}</strong></p>
    `;
}

function updateRulesDisplay(data) {
    if (data.rules && data.rules_options) {
        // Store the rules in the local rules object
        rules = { ...data.rules };

        // Build HTML for rules
        let rulesHTML = '';

        for (const [key, value] of Object.entries(data.rules)) {
            // Format the key for display
            const displayKey = `${key.charAt(0).toUpperCase() + key.slice(1)}:`;

            // Create options HTML
            let optionsHTML = '';
            data.rules_options.forEach(option => {
                const selected = option === value ? 'selected' : '';
                const displayOption = option.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                optionsHTML += `<option value="${option}" ${selected}>${displayOption}</option>`;
            });

            // Create rule item HTML
            rulesHTML += `
                <div class="rule-item">
                    <h3>${displayKey}</h3>
                    <select id="${key}-action">
                        ${optionsHTML}
                    </select>
                </div>
            `;
        }

        // Set the HTML content
        document.querySelector('.rule-container').innerHTML = rulesHTML;
    }
}

function saveRules() {
    // Get the current rules from the UI
    const updatedRules = {};

    document.querySelectorAll('.rule-item').forEach(item => {
        const key = item.querySelector('h3').textContent.replace(':', '').trim().toLowerCase();
        const value = item.querySelector('select').value;
        updatedRules[key] = value;
    });

    // Send rules to server if WebSocket is connected
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            event: 'update_rules',
            data: updatedRules
        }));
        rules = updatedRules;

        // Show brief confirmation
        const saveButton = document.querySelector('.btn-save-rules');
        const originalText = saveButton.textContent;
        saveButton.textContent = "Saved!";
        setTimeout(() => {
            saveButton.textContent = originalText;
        }, 1500);
    }
}

window.controlConveyor = function (state) {
    // Using WebSocket to control conveyor when connected
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            event: 'control_conveyor',
            data: { state: state }
        }));
    }
};