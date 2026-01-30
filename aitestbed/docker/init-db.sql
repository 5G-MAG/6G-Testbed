-- =============================================================================
-- 6G AI Traffic Testbed - PostgreSQL Initialization
-- =============================================================================
-- This script runs when the PostgreSQL container is first created.
-- It sets up sample data for the Data Analyst agent scenario.

-- Create sample tables for testing
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10, 2),
    stock_quantity INTEGER,
    rating DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER,
    total_price DECIMAL(10, 2),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO products (name, category, price, stock_quantity, rating) VALUES
    ('Laptop Pro 15', 'Electronics', 1299.99, 50, 4.5),
    ('Wireless Mouse', 'Electronics', 49.99, 200, 4.2),
    ('USB-C Hub', 'Electronics', 79.99, 150, 4.0),
    ('Mechanical Keyboard', 'Electronics', 149.99, 100, 4.7),
    ('Monitor 27"', 'Electronics', 399.99, 75, 4.3),
    ('Webcam HD', 'Electronics', 89.99, 120, 4.1),
    ('Headphones Pro', 'Audio', 249.99, 80, 4.6),
    ('Portable Speaker', 'Audio', 129.99, 90, 4.4),
    ('Smart Watch', 'Wearables', 299.99, 60, 4.5),
    ('Fitness Tracker', 'Wearables', 99.99, 140, 4.2)
ON CONFLICT DO NOTHING;

INSERT INTO customers (name, email, tier) VALUES
    ('Alice Johnson', 'alice@example.com', 'premium'),
    ('Bob Smith', 'bob@example.com', 'standard'),
    ('Carol Williams', 'carol@example.com', 'premium'),
    ('David Brown', 'david@example.com', 'standard'),
    ('Eve Davis', 'eve@example.com', 'basic')
ON CONFLICT DO NOTHING;

INSERT INTO orders (customer_id, product_id, quantity, total_price, status) VALUES
    (1, 1, 1, 1299.99, 'delivered'),
    (1, 2, 2, 99.98, 'delivered'),
    (2, 4, 1, 149.99, 'shipped'),
    (3, 7, 1, 249.99, 'delivered'),
    (3, 9, 1, 299.99, 'processing'),
    (4, 3, 3, 239.97, 'delivered'),
    (5, 10, 2, 199.98, 'shipped'),
    (1, 5, 1, 399.99, 'delivered'),
    (2, 6, 1, 89.99, 'delivered'),
    (4, 8, 1, 129.99, 'processing')
ON CONFLICT DO NOTHING;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_tier ON customers(tier);

-- Create views for common queries
CREATE OR REPLACE VIEW order_summary AS
SELECT
    c.name as customer_name,
    c.tier as customer_tier,
    p.name as product_name,
    p.category,
    o.quantity,
    o.total_price,
    o.status,
    o.created_at as order_date
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id;

CREATE OR REPLACE VIEW product_stats AS
SELECT
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price,
    SUM(stock_quantity) as total_stock,
    AVG(rating) as avg_rating
FROM products
GROUP BY category;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO testbed;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO testbed;
