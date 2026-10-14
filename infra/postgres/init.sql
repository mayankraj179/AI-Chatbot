-- Executed by the postgres image on first start (empty volume) only.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE DATABASE chatbot_test OWNER chatbot;
\connect chatbot_test
CREATE EXTENSION IF NOT EXISTS vector;
