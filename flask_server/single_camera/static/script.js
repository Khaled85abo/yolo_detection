// Replace Socket.IO with native WebSocket
let socket;
let connectionStatus = 'disconnected';
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 1000;


const saveRulesButton = document.querySelector('.btn-save-rules');

saveRulesButton.addEventListener('click', saveRules);

// Initial connection and rule loading
window.addEventListener('load', function () {
    connectWebSocket();
});

// Rules configuration
let rules = {};



// Save rules to localStorage and send to server
function saveRules() {
    // Get the current rules from the UI
    const updatedRules = {};
    console.log('Saving rules');
    console.log(updatedRules);
    document.querySelectorAll('.rule-item').forEach(item => {
        const key = item.querySelector('h3').textContent.replace(':', '').trim().toLowerCase();
        const value = item.querySelector('select').value;
        updatedRules[key] = value;
    });

    console.log('Updated rules');
    console.log(updatedRules);
    // Send rules to server if WebSocket is connected
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            event: 'update_rules',
            data: updatedRules
        }));
        rules = updatedRules;
        console.log('Rules sent to server:', updatedRules);
    } else {
        console.error('WebSocket not connected, rules saved locally only');
    }
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
        console.log('Raw message received:', event.data);
        try {
            const data = JSON.parse(event.data);
            console.log('Parsed message:', data);

            // Handle different event types
            if (data.event === 'status_update') {
                updateStatusDisplay(data.data);
            }
            if (data.event === 'rules_update') {
                updateRulesDisplay(data.data);
            }
            // Add more event handlers as needed
        } catch (error) {
            console.error('Error parsing message:', error, event.data);
        }
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
        document.getElementById('connection-status').textContent = 'Connection Error';
        document.querySelector('.status-indicator').classList.remove('status-connected', 'status-initializing');
        document.querySelector('.status-indicator').classList.add('status-disconnected');

        // Try to reconnect immediately on error
        if (connectionStatus === 'connected') {
            setTimeout(connectWebSocket, 1000);
        }
    };

    socket.onclose = function (event) {
        console.log('Disconnected from server websocket:', event.reason);
        connectionStatus = 'disconnected';
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.querySelector('.status-indicator').classList.remove('status-connected', 'status-initializing');
        document.querySelector('.status-indicator').classList.add('status-disconnected');

        // Handle reconnection
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = reconnectDelay * Math.min(reconnectAttempts, 5); // Cap the delay multiplication
            console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`);

            setTimeout(connectWebSocket, delay);
        } else {
            console.log('Max reconnection attempts reached');
            document.getElementById('connection-status').textContent = 'Reconnection Failed';
        }
    };
}


function updateRulesDisplay(data) {
    console.log('Updating rules display:', data);
    if (data.rules && data.rules_options) {
        // Check if the received rules are the same as the existing rules
        let rulesChanged = false;

        // Compare each rule value
        for (const [key, value] of Object.entries(data.rules)) {
            if (!rules[key] || rules[key] !== value) {
                rulesChanged = true;
                break;
            }
        }

        // If rules haven't changed, don't update the UI
        if (!rulesChanged) {
            console.log('Rules unchanged, skipping UI update');
            return;
        }

        // Store the rules in the local rules object
        rules = { ...data.rules };

        // Build HTML using template literals
        let rulesHTML = '';

        for (const [key, value] of Object.entries(data.rules)) {
            // Format the key for display (e.g., "overlap" -> "When Overlap Detected:")
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
                    <select id="${key}-action" onchange="updateRule('${key}', this.value)">
                        ${optionsHTML}
                    </select>
                </div>
            `;
        }

        // Set the HTML content
        const ruleContainer = document.querySelector('.rule-container');
        ruleContainer.innerHTML = rulesHTML;
        console.log('Updated rules display with server data');
    }
}
function updateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => updateStatusDisplay(data))
        .catch(err => console.error('Error fetching status:', err));
}
function updateStatusDisplay(data) {
    console.log('Updating status display:', data);
    document.getElementById('status').innerHTML = `
                <p>Conveyor: <strong>${data.conveyor_stop ? '⛔ STOPPED' : '✅ RUNNING'}</strong></p>
                <p>Overlapped: <strong>${data.overlap ? '⚠️ YES' : '✅ NO'}</strong></p>
                <p>Stopped: <strong>${data.stop ? '⚠️ YES' : '✅ NO'}</strong></p>
                <p>Incorrect: <strong>${data.incorrect ? '⚠️ YES' : '✅ NO'}</strong></p>
            `;
}

window.controlConveyor = function (state) {
    console.log("Control conveyor action:", state);

    // Using WebSocket to control conveyor when connected
    if (socket && socket.readyState === WebSocket.OPEN) {
        const message = JSON.stringify({
            event: 'control_conveyor',
            data: { state: state }
        });
        socket.send(message);
        console.log('Sent via WebSocket:', message);
    } else {
        // Fallback to HTTP API if WebSocket is not connected
        console.log('WebSocket not connected, using HTTP API');
        fetch('/api/control_conveyor', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ state: state })
        })
            .then(response => response.json())
            .then(data => {
                console.log('Control conveyor response:', data);
                updateStatus();
            })
            .catch(err => console.error('Error controlling conveyor:', err));
    }
};