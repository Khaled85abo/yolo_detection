const video = document.getElementById('video');
const status = document.getElementById('status');

async function start() {
    try {
        const pc = new RTCPeerConnection({
            sdpSemantics: 'unified-plan',
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        pc.addEventListener('track', (evt) => {
            if (evt.track.kind === 'video') {
                video.srcObject = evt.streams[0];
            }
        });

        // Create offer
        const offer = await pc.createOffer({
            offerToReceiveVideo: true,
            offerToReceiveAudio: false
        });
        await pc.setLocalDescription(offer);

        // Send offer to server
        const response = await fetch('/offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
                camera_id: 'camera1'
            })
        });

        const answer = await response.json();
        await pc.setRemoteDescription(answer);
        status.textContent = 'Status: Connected';

        // Handle connection state changes
        pc.addEventListener('connectionstatechange', () => {
            status.textContent = `Status: ${pc.connectionState}`;
            if (pc.connectionState === 'failed') {
                pc.close();
                setTimeout(start, 1000);  // Try to reconnect after 1 second
            }
        });

    } catch (e) {
        console.error('Error starting WebRTC:', e);
        status.textContent = `Status: Error - ${e.message}`;
        setTimeout(start, 1000);  // Try to reconnect after 1 second
    }
}

start();