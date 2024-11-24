import fetch from 'node-fetch';  // Need to import fetch for Node.js

async function getAiResponse(prompt) {
    try {
        // Step 1: Create Chat Session
        const sessionResponse = await fetch('https://api.on-demand.io/chat/v1/sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': 'ZkNPJIOGJ6CHcdT1PaXhnIcdcSVwcKB2',
            },
            body: JSON.stringify({
                "pluginIds": ["plugin-1717464304"],
                "externalUserId": "Phase",
            }),
        });

        if (!sessionResponse.ok) {
            throw new Error(`Failed to create session: ${sessionResponse.status} ${sessionResponse.statusText}`);
        }

        const sessionData = await sessionResponse.json();
        const sessionId = sessionData.data.id;

        // Step 2: Answer Query using session ID from Step 1
        const queryResponse = await fetch(`https://api.on-demand.io/chat/v1/sessions/${sessionId}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': 'ZkNPJIOGJ6CHcdT1PaXhnIcdcSVwcKB2',
            },
            body: JSON.stringify({
                "endpointId": "predefined-openai-gpt4o",
                "query": prompt,
                "pluginIds": ["plugin-1717464304"],
                "responseMode": "sync",
            }),
        });

        if (!queryResponse.ok) {
            throw new Error(`Failed to get query response: ${queryResponse.status} ${queryResponse.statusText}`);
        }

        const queryData = await queryResponse.json();
        
        if (!queryData.data || !queryData.data.answer) {
            throw new Error('Invalid response format from AI service');
        }

        return queryData.data.answer;
    } catch (error) {
        console.error('Error in getAiResponse:', error);
        throw error; // Propagate error to be handled by the server
    }
}

export { getAiResponse };