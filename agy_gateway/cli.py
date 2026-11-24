"""Command-line interface for running agy-api-gateway and managing keys & accounts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional

import uvicorn

from agy_gateway.accounts import AccountStore
from agy_gateway.agy_client import AgyRunner
from agy_gateway.config import get_settings
from agy_gateway.db import migrate_sqlite_to_postgres
from agy_gateway.dispatcher import AgyDispatcher
from agy_gateway.housekeeping import prune_agy_artifacts
from agy_gateway.keys import KeyStore


def _cmd_serve(args: argparse.Namespace) -> int:
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    uvicorn.run(
        "agy_gateway.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def _cmd_key_create(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = KeyStore(settings.db_path, database_url=settings.database_url)
    allowed_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else []
    )
    allowed_accounts = (
        [a.strip() for a in args.accounts.split(",") if a.strip()]
        if args.accounts
        else []
    )
    record, raw_key = store.create_key(
        name=args.name,
        rpm_limit=args.rpm,
        allowed_models=allowed_models,
        allowed_accounts=allowed_accounts,
    )
    base_url = f"http://{settings.host}:{settings.port}"
    print(f"Created API key for '{record.name}' (id={record.id})")
    print(f"API Key (shown once): {raw_key}")
    print(f"OpenAI base_url:      {base_url}/v1")
    print(f"Gemini base_url:      {base_url}/v1beta")
    print(f"Anthropic base_url:   {base_url}")
    print(f"RPM limit:            {record.rpm_limit or 'unlimited'}")
    if record.allowed_models:
        print(f"Allowed models:       {', '.join(record.allowed_models)}")
    if record.allowed_accounts:
        print(f"Allowed accounts:     {', '.join(record.allowed_accounts)}")
    return 0


def _cmd_key_list(_: argparse.Namespace) -> int:
    settings = get_settings()
    store = KeyStore(settings.db_path, database_url=settings.database_url)
    keys = store.list_keys()
    if not keys:
        print("No API keys issued yet.")
        return 0
    for rec in keys:
        status = "enabled" if rec.enabled else "disabled"
        models = ",".join(rec.allowed_models) if rec.allowed_models else "*"
        accs = ",".join(rec.allowed_accounts) if rec.allowed_accounts else "*"
        print(
            f"{rec.id}  {rec.name:<20} {rec.key_prefix:<18} "
            f"[{status}] rpm={rec.rpm_limit} models={models} accounts={accs} "
            f"reqs={rec.total_requests} in={rec.total_input_tokens} out={rec.total_output_tokens}"
        )
    return 0


def _cmd_key_toggle(args: argparse.Namespace, enabled: bool) -> int:
    settings = get_settings()
    store = KeyStore(settings.db_path, database_url=settings.database_url)
    ok = store.set_enabled(args.key_id, enabled=enabled)
    if not ok:
        print(f"Error: key '{args.key_id}' not found", file=sys.stderr)
        return 1
    state = "enabled" if enabled else "disabled"
    print(f"Key {args.key_id} {state}.")
    return 0


def _cmd_key_delete(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = KeyStore(settings.db_path, database_url=settings.database_url)
    ok = store.delete_key(args.key_id)
    if not ok:
        print(f"Error: key '{args.key_id}' not found", file=sys.stderr)
        return 1
    print(f"Deleted key {args.key_id}.")
    return 0


def _cmd_account_list(_: argparse.Namespace) -> int:
    settings = get_settings()
    store = AccountStore(
        settings.db_path, settings.accounts_dir, database_url=settings.database_url
    )
    print(f"Routing strategy: {store.get_routing_strategy(settings.routing_strategy)}")
    for acc in store.list_accounts():
        flag = "enabled" if acc.enabled else "disabled"
        print(
            f"{acc.id:<16} {acc.name:<18} email={acc.email or '-':<26} "
            f"[{flag}/{acc.status}] prio={acc.priority} max_conc={acc.max_concurrency} "
            f"reqs={acc.total_requests} errs={acc.total_errors}"
        )
    return 0


def _cmd_account_create(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = AccountStore(
        settings.db_path, settings.accounts_dir, database_url=settings.database_url
    )
    acc = store.create_account(
        name=args.name,
        priority=args.priority,
        max_concurrency=args.max_concurrency,
    )
    print(f"Created account '{acc.name}' (id={acc.id}, home={acc.home_dir})")
    print("Sign in via the Web UI (/ui -> 账号池) or run:")
    print(f"  HOME={acc.home_dir} agy acp --login")
    return 0


def _cmd_account_check(args: argparse.Namespace) -> int:
    settings = get_settings()
    dispatcher = AgyDispatcher(settings)
    updated, models, err = asyncio.run(dispatcher.probe_and_update_account(args.account_id))
    print(f"Account {updated.id}: status={updated.status} email={updated.email or '-'} models={len(models)}")
    if err:
        print(f"Detail: {err}")
    return 0 if updated.status == "logged_in" else 1


def _cmd_account_delete(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = AccountStore(
        settings.db_path, settings.accounts_dir, database_url=settings.database_url
    )
    try:
        ok = store.delete_account(args.account_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not ok:
        print(f"Error: account '{args.account_id}' not found", file=sys.stderr)
        return 1
    print(f"Deleted account {args.account_id}.")
    return 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = AccountStore(
        settings.db_path, settings.accounts_dir, database_url=settings.database_url
    )
    from agy_gateway.sessions import SessionStore

    sess_store = SessionStore(
        settings.db_path,
        ttl_hours=settings.session_ttl_hours,
        database_url=settings.database_url,
    )
    sess_store.prune_expired()
    protected = sess_store.list_active_conversation_ids()
    homes = [a.resolved_home() for a in store.list_accounts()]
    report = prune_agy_artifacts(
        homes,
        retention_minutes=args.retention_minutes,
        workdir=settings.workdir,
        protected_conversation_ids=protected,
    )
    freed_mb = report.freed_bytes / (1024 * 1024)
    print(
        f"Pruned {report.removed_entries} stale entries (preserved {len(protected)} active sessions), "
        f"freed {freed_mb:.2f} MiB."
    )
    return 0


def _cmd_db_migrate(args: argparse.Namespace) -> int:
    settings = get_settings()
    src = args.sqlite or str(settings.db_path)
    target = args.target_url or settings.database_url
    if not target:
        print(
            "Error: provide --target-url or set AGY_GATEWAY_DATABASE_URL",
            file=sys.stderr,
        )
        return 1
    report = migrate_sqlite_to_postgres(
        sqlite_path=src,
        database_url=target,
        accounts_root=settings.accounts_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    settings = get_settings()
    runner = AgyRunner(
        agy_bin=settings.agy_bin,
        workdir=settings.workdir,
        sandbox=settings.sandbox,
        disable_slash_commands=settings.disable_slash_commands,
    )
    models = asyncio.run(runner.list_models(force_refresh=True))
    print("Available Antigravity models:")
    for m in models:
        print(f"  {m.id:<28} {m.display_name}")
    print("\nConfigured model aliases:")
    print(json.dumps(settings.model_aliases, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agy-gateway",
        description="API gateway dispatching requests to Google Antigravity CLI (agy).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the gateway HTTP server")
    serve_p.add_argument("--host", default=None, help="Bind host (default from settings)")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port (default 8787)")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_p.set_defaults(func=_cmd_serve)

    models_p = sub.add_parser("models", help="List available Antigravity models & aliases")
    models_p.set_defaults(func=_cmd_models)

    cleanup_p = sub.add_parser("cleanup", help="Prune old stateless agy conversation logs/DBs")
    cleanup_p.add_argument("--retention-minutes", type=int, default=30)
    cleanup_p.set_defaults(func=_cmd_cleanup)

    key_p = sub.add_parser("key", help="Manage issued API keys")
    key_sub = key_p.add_subparsers(dest="key_cmd", required=True)

    kc = key_sub.add_parser("create", help="Create a new API key")
    kc.add_argument("--name", required=True, help="Friendly label for the client/harness")
    kc.add_argument("--rpm", type=int, default=30, help="Requests/minute limit (0=unlimited)")
    kc.add_argument("--models", default="", help="Comma-separated allowed model IDs")
    kc.add_argument("--accounts", default="", help="Comma-separated allowed account IDs")
    kc.set_defaults(func=_cmd_key_create)

    kl = key_sub.add_parser("list", help="List all issued API keys")
    kl.set_defaults(func=_cmd_key_list)

    kd = key_sub.add_parser("disable", help="Disable an API key")
    kd.add_argument("key_id", help="Key ID (e.g. key_abc123...)")
    kd.set_defaults(func=lambda a: _cmd_key_toggle(a, False))

    ke = key_sub.add_parser("enable", help="Re-enable an API key")
    ke.add_argument("key_id", help="Key ID (e.g. key_abc123...)")
    ke.set_defaults(func=lambda a: _cmd_key_toggle(a, True))

    krm = key_sub.add_parser("delete", help="Delete an API key permanently")
    krm.add_argument("key_id", help="Key ID (e.g. key_abc123...)")
    krm.set_defaults(func=_cmd_key_delete)

    acc_p = sub.add_parser("account", help="Manage Antigravity account pool")
    acc_sub = acc_p.add_subparsers(dest="acc_cmd", required=True)

    al = acc_sub.add_parser("list", help="List configured Antigravity accounts")
    al.set_defaults(func=_cmd_account_list)

    ac = acc_sub.add_parser("create", help="Create a new isolated Antigravity account slot")
    ac.add_argument("--name", required=True, help="Display name for the account")
    ac.add_argument("--priority", type=int, default=100, help="Lower number = higher priority")
    ac.add_argument("--max-concurrency", type=int, default=6, help="Per-account concurrency cap")
    ac.set_defaults(func=_cmd_account_create)

    achk = acc_sub.add_parser("check", help="Check login health of an account")
    achk.add_argument("account_id", help="Account ID (e.g. default or acc_...)")
    achk.set_defaults(func=_cmd_account_check)

    arm = acc_sub.add_parser("delete", help="Delete an account slot and its isolated profile")
    arm.add_argument("account_id", help="Account ID")
    arm.set_defaults(func=_cmd_account_delete)

    db_p = sub.add_parser("db", help="Database maintenance and migration utilities")
    db_sub = db_p.add_subparsers(dest="db_cmd", required=True)
    dbm = db_sub.add_parser("migrate", help="Idempotently migrate SQLite data to PostgreSQL")
    dbm.add_argument("--sqlite", default="", help="Path to source SQLite DB (default from settings)")
    dbm.add_argument("--target-url", default="", help="Target PostgreSQL DSN (default from AGY_GATEWAY_DATABASE_URL)")
    dbm.set_defaults(func=_cmd_db_migrate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
