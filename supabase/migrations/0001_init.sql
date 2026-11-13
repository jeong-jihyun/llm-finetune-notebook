-- llm-finetune-notebook 초기 스키마
create extension if not exists vector;

create table documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  storage_path text,
  status text not null default 'pending' check (status in ('pending','processing','ready','failed')),
  error text,
  created_at timestamptz not null default now()
);

create table chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  idx int not null,
  content text not null,
  embedding vector(1536)
);
create index chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index chunks_user_idx on chunks(user_id);

create table training_examples (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  chunk_id uuid references chunks(id) on delete set null,
  question text not null,
  answer text not null,
  approved boolean not null default true,
  created_at timestamptz not null default now()
);

create table finetune_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null,
  provider_job_id text,
  base_model text not null,
  status text not null default 'queued' check (status in ('queued','running','succeeded','failed','cancelled')),
  example_count int not null default 0,
  error text,
  created_at timestamptz not null default now(),
  finished_at timestamptz
);

create table models (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  job_id uuid references finetune_jobs(id) on delete set null,
  name text not null,
  provider_model_id text not null,
  created_at timestamptz not null default now()
);

create table messages (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user','assistant')),
  content text not null,
  model_id uuid references models(id) on delete set null,
  sources jsonb,
  created_at timestamptz not null default now()
);

-- 본인 데이터만 접근 (RLS)
do $$
declare t text;
begin
  foreach t in array array['documents','chunks','training_examples','finetune_jobs','models','messages']
  loop
    execute format('alter table %I enable row level security', t);
    execute format('create policy "own rows" on %I for all using (user_id = auth.uid()) with check (user_id = auth.uid())', t);
  end loop;
end $$;

-- 비슷한 청크 검색 함수
create or replace function match_chunks(query_embedding vector(1536), match_count int default 5)
returns table (id uuid, document_id uuid, content text, similarity float)
language sql stable as $$
  select c.id, c.document_id, c.content, 1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  where c.user_id = auth.uid()
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
