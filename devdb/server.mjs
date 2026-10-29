// Local Postgres + pgvector for development and tests, with no Docker or install:
// PGlite (Postgres compiled to WebAssembly) served over the normal Postgres wire protocol.
//   node devdb/server.mjs            -> postgresql://postgres@127.0.0.1:5433/postgres  (in-memory)
//   PGLITE_DATA=./pgdata node ...    -> same, persisted to a folder
// In CI and production, point RECALL_PG_DSN at any real Postgres with pgvector instead.
import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { PGLiteSocketServer } from '@electric-sql/pglite-socket'

const port = Number(process.env.PGLITE_PORT || 5433)
const db = await PGlite.create(process.env.PGLITE_DATA || undefined, { extensions: { vector } })
await db.exec('CREATE EXTENSION IF NOT EXISTS vector;')
const server = new PGLiteSocketServer({ db, port, host: '127.0.0.1', maxConnections: 8 })
await server.start()
console.log(`ready postgresql://postgres@127.0.0.1:${port}/postgres`)
const stop = async () => { await server.stop(); await db.close(); process.exit(0) }
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
