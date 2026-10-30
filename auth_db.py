import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash, check_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "healthguard_users.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create account and password-reset tables if they do not exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('user', 'admin')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                disease_type TEXT NOT NULL,
                prediction_result TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
                    ON DELETE CASCADE
        );
            """
        )


def create_user(name, email, password):
    """Create a user account with a securely hashed password."""
    name = name.strip()
    email = email.strip().lower()

    password_hash = generate_password_hash(password)

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (name, email, password_hash),
            )

            return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None


def get_user_by_email(email):
    """Find a user by email address."""
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()


def get_user_by_id(user_id):
    """Find a user by their database ID."""
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, name, email, role, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()


def verify_password(user, password):
    """Verify a password against the stored hash."""
    if user is None:
        return False

    return check_password_hash(user["password_hash"], password)


def update_profile(user_id, name, email):
    """Update the user's name and email."""
    try:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET name = ?, email = ?
                WHERE id = ?
                """,
                (name.strip(), email.strip().lower(), user_id),
            )

        return True

    except sqlite3.IntegrityError:
        return False


def change_password(user_id, new_password):
    """Save a new hashed password."""
    password_hash = generate_password_hash(new_password)

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (password_hash, user_id),
        )

def get_all_users():
    """Return account details without password hashes."""
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, name, email, role, created_at
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()

def get_all_users():
    """Return all users for the administrator dashboard."""
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, name, email, role, created_at
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()

def create_password_reset_token(user_id, token_hash, expires_at):
    """Store a hashed password-reset token and its expiration time."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO password_reset_tokens
                (user_id, token_hash, expires_at, used)
            VALUES (?, ?, ?, 0)
            """,
            (user_id, token_hash, expires_at),
        )


def get_password_reset_token(token_hash):
    """Retrieve a reset token and its associated user."""
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, user_id, token_hash, expires_at, used
            FROM password_reset_tokens
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

def mark_password_reset_token_used(token_id):
    """Mark a reset token as used so it cannot be reused."""
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE password_reset_tokens
            SET used = 1
            WHERE id = ? AND used = 0
            """,
            (token_id,),
        )


def reset_user_password(user_id, new_password):
    """Update a user's password with a securely hashed password."""
    password_hash = generate_password_hash(new_password)

    with get_connection() as connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )

def save_prediction(user_id, disease_type, prediction_result):
    """Save a prediction result for a specific user."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
                (user_id, disease_type, prediction_result)
            VALUES (?, ?, ?)
            """,
            (user_id, disease_type, prediction_result),
        )


def get_user_prediction_history(user_id):
    """Retrieve prediction history belonging only to a specific user."""
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, disease_type, prediction_result, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()