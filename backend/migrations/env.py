"""
Alembic environment.

The URL comes from settings.DATABASE_URL (i.e. the DATABASE_URL environment
variable, or the SQLite default beside the source) rather than from alembic.ini,
so migrations always target the same database the application does and no path
or credential is written into a tracked file.

`render_as_batch` is on because SQLite cannot ALTER most things in place —
Alembic emits a copy-and-swap instead, which is what makes a future column
change actually runnable on the deployed database rather than only on Postgres.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import settings
from models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations against the live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            # Without this, SQLite ignores the ON DELETE CASCADE on
            # face_template.person_id during a migration run.
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
