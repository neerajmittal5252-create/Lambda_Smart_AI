create extension if not exists vector;

create table if not exists chat_messages (
    id uuid default gen_random_uuid() primary key,
    thread_id text not null,
    role text not null,
    content text,
    created_at timestamp with time zone default now()
);

create table if not exists repo_embeddings (
    id uuid default gen_random_uuid() primary key,
    content text not null,
    embedding vector(1536),
    metadata jsonb default '{}',
    created_at timestamp with time zone default now()
);

create or replace function match_repo_embeddings(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
returns table(
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        id,
        content,
        metadata,
        1 - (repo_embeddings.embedding <=> query_embedding) as similarity
    from repo_embeddings
    where 1 - (repo_embeddings.embedding <=> query_embedding) > match_threshold
    order by repo_embeddings.embedding <=> query_embedding
    limit match_count;
$$;