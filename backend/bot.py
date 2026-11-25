
import logging
import os

from dotenv import load_dotenv

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN belum ditemukan. "
        "Periksa file .env."
    )


# ==========================================
# 2. LOGGING
# ==========================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ==========================================
# 3. MAIN MENU
# ==========================================

MAIN_MENU = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton("Informasi Akademik"),
            KeyboardButton("Kalender Akademik"),
        ],
        [
            KeyboardButton("Layanan Administrasi"),
            KeyboardButton("Bantuan"),
        ],
    ],
    resize_keyboard=True,
    input_field_placeholder="Tulis pertanyaan akademik..."
)


# ==========================================
# 4. START COMMAND
# ==========================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user = update.effective_user

    name = user.first_name if user else "Mahasiswa"

    welcome_message = (
        f"Halo, {name}! 👋\n\n"
        "Selamat datang di Q-TIME.\n\n"
        "Saya adalah asisten digital yang dirancang "
        "untuk membantu kebutuhan informasi akademik "
        "dan layanan mahasiswa.\n\n"
        "Silakan pilih menu di bawah ini atau "
        "langsung tuliskan pertanyaan Anda.\n\n"
        "Saat ini sistem sedang dalam tahap pengembangan."
    )

    await update.effective_message.reply_text(
        text=welcome_message,
        reply_markup=MAIN_MENU
    )


# ==========================================
# 5. HELP COMMAND
# ==========================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    help_message = (
        "📚 PETUNJUK PENGGUNAAN\n\n"
        "/start - Memulai bot\n"
        "/help - Melihat bantuan\n\n"
        "Anda dapat menggunakan menu yang tersedia "
        "atau mengirimkan pertanyaan secara langsung.\n\n"
        "Contoh:\n"
        "• Bagaimana prosedur pengajuan cuti?\n"
        "• Kapan jadwal pengisian KRS?\n"
        "• Apa persyaratan pendaftaran wisuda?\n\n"
        "Fitur AI akan diintegrasikan pada milestone berikutnya."
    )

    await update.effective_message.reply_text(
        text=help_message,
        reply_markup=MAIN_MENU
    )


# ==========================================
# 6. TEXT MESSAGE HANDLER
# ==========================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    message = update.effective_message

    if message is None or message.text is None:
        return

    user_message = message.text.strip()

    logger.info("Incoming text message received.")

    normalized_message = user_message.casefold()

    responses = {
        "informasi akademik": (
            "📚 Informasi Akademik\n\n"
            "Menu informasi akademik sedang disiapkan "
            "untuk integrasi knowledge base kampus."
        ),

        "kalender akademik": (
            "📅 Kalender Akademik\n\n"
            "Informasi kalender akademik akan "
            "tersedia setelah dokumen resmi kampus "
            "diintegrasikan."
        ),

        "layanan administrasi": (
            "📄 Layanan Administrasi\n\n"
            "Panduan administrasi akademik akan "
            "tersedia pada pengembangan berikutnya."
        ),

        "bantuan": (
            "Silakan tuliskan pertanyaan Anda "
            "atau gunakan menu layanan yang tersedia."
        ),
    }

    response = responses.get(
        normalized_message,
        (
            "✅ Pesan Anda berhasil diterima!\n\n"
            "Saat ini Q-TIME masih "
            "dalam tahap pengembangan konektivitas.\n\n"
            "Kemampuan memahami pertanyaan akademik "
            "menggunakan AI akan ditambahkan pada "
            "milestone berikutnya."
        )
    )

    await message.reply_text(
        text=response,
        reply_markup=MAIN_MENU
    )


# ==========================================
# 7. ERROR HANDLER
# ==========================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):
    logger.error(
        "An error occurred while processing an update.",
        exc_info=context.error
    )


# ==========================================
# 8. BOT INITIALIZATION
# ==========================================

async def post_init(application: Application):
    bot_info = await application.bot.get_me()

    logger.info(
        "Bot berhasil terhubung: @%s",
        bot_info.username
    )


def main():

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Register command handlers
    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        CommandHandler("help", help_command)
    )

    # Register text handler
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    # Register error handler
    application.add_error_handler(error_handler)

    logger.info("Starting Q-TIME")

    # Local development: long polling
    application.run_polling()


if __name__ == "__main__":
    main()
