# 🤖 Telegram Integration Setup Guide

This guide will help you set up Telegram notifications for your ATM Insights monitoring system.

## Prerequisites

- A Telegram account
- Admin access to your ATM Insights backend configuration
- Basic command line knowledge

## Step 1: Create a Telegram Bot

1. **Open Telegram** and search for `@BotFather`
2. **Start a conversation** with BotFather by clicking "START"
3. **Create a new bot** by sending the command:
   ```
   /newbot
   ```
4. **Choose a name** for your bot (e.g., "ATM Insights Monitor")
5. **Choose a username** for your bot (must end with 'bot', e.g., "atm_insights_monitor_bot")
6. **Save the bot token** - BotFather will provide you with a token that looks like:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

## Step 2: Get Your Chat ID

### Option A: Using a Telegram Bot (Recommended)

1. **Search for `@userinfobot`** in Telegram
2. **Start the bot** and it will immediately send you your chat ID
3. **Copy the ID** - it will look like a number (e.g., `123456789`)

### Option B: Using Telegram Web/Desktop

1. **Open Telegram Web** (web.telegram.org) or use Telegram Desktop
2. **Start a chat** with your newly created bot
3. **Send any message** to your bot (e.g., "Hello")
4. **Get your chat ID** by visiting this URL in your browser:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   Replace `<YOUR_BOT_TOKEN>` with your actual bot token
5. **Find your chat ID** in the JSON response under `message.chat.id`

### Option C: Using curl (Command Line)

```bash
# Send a message to your bot first, then run:
curl -s "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates" | grep -o '"chat":{"id":[0-9]*' | grep -o '[0-9]*'
```

## Step 3: Configure Your ATM Insights Backend

### Environment Variables

Add the following environment variables to your `.env` file:

```env
# Telegram Configuration
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### Docker Environment (if using Docker)

Add to your `docker-compose.yml` file under the backend service:

```yaml
backend:
  environment:
    - TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
    - TELEGRAM_CHAT_ID=123456789
```

## Step 4: Test Your Configuration

### Using the API Test Endpoint

1. **Start your ATM Insights backend**
2. **Access the API documentation** at `http://localhost:8000/docs`
3. **Find the test notifications endpoint**: `/api/v1/alerts/notifications/test`
4. **Send a test request** with the following payload:
   ```json
   {
     "channels": ["telegram"],
     "test_message": "Test notification from ATM Insights"
   }
   ```

### Using curl

```bash
curl -X POST "http://localhost:8000/api/v1/alerts/notifications/test" \
     -H "Content-Type: application/json" \
     -d '{
       "channels": ["telegram"],
       "test_message": "Test notification from ATM Insights"
     }'
```

## Step 5: Verify Bot Permissions

Ensure your bot has the following permissions:

1. **Send messages** - Should be enabled by default
2. **Send photos** (optional) - For future image attachments
3. **Send documents** (optional) - For future report attachments

## Alert Notification Behavior

With the current configuration:

- **All Alerts** → Sent to Telegram 📱
- **Critical Alerts** → Sent to both Telegram AND Email 📧📱

### Alert Severity Levels:

- 🚨 **Critical** - Immediate action required (Telegram + Email)
- ⚠️ **High** - Urgent attention needed (Telegram only)
- ⚡ **Medium** - Should be addressed soon (Telegram only)
- ℹ️ **Low** - Informational (Telegram only)

## Troubleshooting

### Bot Token Issues

**Error**: `Telegram bot not configured`

- **Solution**: Check that `TELEGRAM_BOT_TOKEN` is set correctly
- **Verify**: Token should be in format `NNNNNNNNNN:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

### Chat ID Issues

**Error**: `Telegram chat ID not configured`

- **Solution**: Ensure `TELEGRAM_CHAT_ID` is set to your actual chat ID
- **Note**: Chat ID should be a number, not your username

### Message Not Received

1. **Check bot status**: Make sure you've started a conversation with your bot
2. **Verify chat ID**: Use the methods above to double-check your chat ID
3. **Check logs**: Look at backend logs for Telegram-related errors
4. **Network issues**: Ensure your server can reach `api.telegram.org`

### Rate Limiting

Telegram has rate limits:

- **Individual chats**: 30 messages per second
- **Broadcast**: 30 messages per second across all chats

The system includes cooldown periods to prevent spam.

## Group Chat Setup (Optional)

To send alerts to a group chat:

1. **Create a Telegram group**
2. **Add your bot** to the group
3. **Make the bot an admin** (recommended)
4. **Get the group chat ID**:
   - Send a message in the group
   - Use the same methods as above, but the chat ID will be negative (e.g., `-123456789`)
5. **Update your configuration** with the group chat ID

## Security Considerations

- **Keep your bot token secret** - Never commit it to version control
- **Use environment variables** - Store sensitive data in `.env` files
- **Restrict bot permissions** - Only give necessary permissions
- **Monitor bot usage** - Check logs for unusual activity

## Advanced Configuration

### Multiple Chat IDs

To send alerts to multiple chats, you'll need to modify the notification service to support arrays of chat IDs.

### Custom Message Formatting

The Telegram messages use HTML formatting. You can customize the templates in:
`backend/services/notification_service.py` → `_generate_telegram_message()`

### Webhook Setup (Production)

For production environments, consider setting up Telegram webhooks instead of polling for better performance.

## Support

If you encounter issues:

1. **Check the logs** - Backend logs contain detailed error information
2. **Verify configuration** - Double-check all environment variables
3. **Test step by step** - Use the test endpoint to isolate issues
4. **Consult Telegram API docs** - https://core.telegram.org/bots/api

---

🎉 **Congratulations!** Your ATM Insights system is now configured to send real-time alerts to Telegram!
