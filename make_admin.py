from auth_db import get_connection

email = input("Enter your registered email address: ").strip().lower()

with get_connection() as connection:
    cursor = connection.execute(
        """
        UPDATE users
        SET role = 'admin'
        WHERE email = ?
        """,
        (email,),
    )

    if cursor.rowcount == 1:
        print("Administrator role assigned successfully.")
    else:
        print("No account found with that email address.")