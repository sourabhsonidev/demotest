import logging
from flask import request, jsonify
from datetime import datetime
from pr_reviewer.config.github_config import GitHubAppConfig
from pr_reviewer.utils.misc_utils import is_owner_whitelisted
from pr_reviewer.github_app.webhook_handlers import (
    verify_webhook_signature,
    handle_pull_request_event,
    handle_pull_request_review_event,
    handle_pull_request_review_comment_event
)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_webhook():
    """Handle GitHub webhook events"""
    
    # Verify webhook signature
    signature = request.headers.get('X-Hub-Signature-256')
    if not signature or not verify_webhook_signature(
        request.data, 
        signature, 
        GitHubAppConfig.GITHUB_WEBHOOK_SECRET
    ):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Parse webhook payload
    event_type = request.headers.get('X-GitHub-Event')
    payload = request.json

    if event_type and event_type.startswith("pull_request"):
        if not is_owner_whitelisted(payload):
            logger.warning(
                "Webhook rejected: owner not whitelisted",
                extra={
                    "owner": payload.get("repository", {}).get("owner", {}).get("login"),
                    "repo": payload.get("repository", {}).get("full_name")
                }
            )
            return jsonify({"message": "Owner not allowed"}), 403

    # Handle different event types
    if event_type == 'pull_request':
        return handle_pull_request_event(payload)
    elif event_type == 'pull_request_review':
        return handle_pull_request_review_event(payload)
    elif event_type == 'pull_request_review_comment':
        logger.info("Handling pull_request_review_comment event")
        return handle_pull_request_review_comment_event(payload)
    else:
        logger.info(f"Unhandled event type: {event_type}")
        return jsonify({'message': 'Event received'}), 200

def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

def app_info():
    """Basic info endpoint"""
    return jsonify({
        'message': 'GitHub App is running',
        'app_id': GitHubAppConfig.GITHUB_APP_ID,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200
